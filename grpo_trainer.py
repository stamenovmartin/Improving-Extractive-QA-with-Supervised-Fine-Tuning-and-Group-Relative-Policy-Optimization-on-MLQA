import copy
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import json

import config
from rewards import compute_reward, normalize_rewards_zscore


class GRPOTrainer:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

        # замрзнат reference model за KL пенал
        self.ref_model = copy.deepcopy(model)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.GRPO_LR)
        self.scaler = torch.amp.GradScaler(enabled=config.USE_AMP)

    @staticmethod
    def build_prompt(question, context):
        return (
            f"Answer the question based on the context.\n\n"
            f"Context: {context}\n\nQuestion: {question}"
        )

    @torch.no_grad()
    def generate_group(self, prompt, K=config.GRPO_GROUP_SIZE):
        enc = self.tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=config.MAX_INPUT_LENGTH,
        ).to(config.DEVICE)

        outputs = self.model.generate(
            **enc,
            max_new_tokens=config.MAX_OUTPUT_LENGTH,
            do_sample=True,
            temperature=config.GRPO_TEMPERATURE,
            top_p=config.GRPO_TOP_P,
            num_return_sequences=K,
            output_scores=False,
            return_dict_in_generate=True,
        )

        results = []
        for seq in outputs.sequences:
            token_ids = seq[1:]
            text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            results.append((text, token_ids))
        return results

    def _sequence_log_probs(self, model, input_ids, attention_mask, target_ids):
        # teacher-forcing: decoder_input_ids = [start] + target[:-1]
        dec_start = torch.full(
            (target_ids.size(0), 1),
            model.config.decoder_start_token_id,
            dtype=torch.long, device=config.DEVICE,
        )
        decoder_input_ids = torch.cat([dec_start, target_ids[:, :-1]], dim=1)

        with torch.amp.autocast(device_type=config.DEVICE, enabled=config.USE_AMP):
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
            ).logits

        # ВАЖНО: log_softmax мора на forward logits, не torch.tensor() wrapper
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(
            2, target_ids.unsqueeze(-1)
        ).squeeze(-1)

        mask = (target_ids != self.tokenizer.pad_token_id).float()
        return (token_log_probs * mask).sum(dim=-1)

    def _per_token_kl(self, input_ids, attention_mask, target_ids):
        dec_start = torch.full(
            (target_ids.size(0), 1),
            self.model.config.decoder_start_token_id,
            dtype=torch.long, device=config.DEVICE,
        )
        decoder_input_ids = torch.cat([dec_start, target_ids[:, :-1]], dim=1)

        with torch.amp.autocast(device_type=config.DEVICE, enabled=config.USE_AMP):
            policy_logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
            ).logits

        policy_log_probs = F.log_softmax(policy_logits, dim=-1)
        policy_probs = policy_log_probs.exp()

        with torch.no_grad():
            ref_logits = self.ref_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
            ).logits
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)

        kl = (policy_probs * (policy_log_probs - ref_log_probs)).sum(dim=-1)
        mask = (target_ids != self.tokenizer.pad_token_id).float()
        per_seq_kl = (kl * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
        return per_seq_kl

    def train_step(self, batch):
        self.model.train()
        total_policy_loss = 0.0
        total_kl = 0.0
        total_reward = 0.0
        clip_count = 0
        total_ratio_count = 0
        n_prompts = 0

        for sample in batch:
            question = sample["question"]
            context = sample["context"]
            answer = sample["answer"]
            prompt = self.build_prompt(question, context)

            group = self.generate_group(prompt, config.GRPO_GROUP_SIZE)
            if not group:
                continue

            texts = [g[0] for g in group]
            token_ids_list = [g[1] for g in group]

            rewards = []
            for txt, tids in zip(texts, token_ids_list):
                r = compute_reward(txt, answer, num_tokens=len(tids))
                rewards.append(r)
            advantages = normalize_rewards_zscore(rewards)

            total_reward += sum(rewards) / len(rewards)

            enc = self.tokenizer(
                prompt, return_tensors="pt", truncation=True,
                max_length=config.MAX_INPUT_LENGTH,
            ).to(config.DEVICE)
            input_ids = enc["input_ids"]
            attention_mask = enc["attention_mask"]

            for i, (txt, tids) in enumerate(zip(texts, token_ids_list)):
                adv = advantages[i]
                if abs(adv) < 1e-8:
                    continue

                tids_tensor = tids[:config.MAX_OUTPUT_LENGTH].unsqueeze(0).to(config.DEVICE)

                # old log prob (detach за stable ratio)
                with torch.no_grad():
                    old_log_prob = self._sequence_log_probs(
                        self.model, input_ids, attention_mask, tids_tensor
                    ).detach()

                new_log_prob = self._sequence_log_probs(
                    self.model, input_ids, attention_mask, tids_tensor
                )

                ratio = torch.exp(new_log_prob - old_log_prob)
                adv_tensor = torch.tensor(adv, device=config.DEVICE, dtype=torch.float32)

                # clipped surrogate (PPO-style)
                surr1 = ratio * adv_tensor
                surr2 = torch.clamp(
                    ratio, 1.0 - config.GRPO_CLIP_EPS, 1.0 + config.GRPO_CLIP_EPS
                ) * adv_tensor
                policy_loss = -torch.min(surr1, surr2).mean()

                kl = self._per_token_kl(input_ids, attention_mask, tids_tensor)
                kl_loss = config.GRPO_KL_BETA * kl.mean()

                loss = policy_loss + kl_loss

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), config.GRPO_MAX_GRAD_NORM
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_policy_loss += policy_loss.item()
                total_kl += kl.item()
                clipped = (ratio < 1 - config.GRPO_CLIP_EPS) | (ratio > 1 + config.GRPO_CLIP_EPS)
                clip_count += clipped.sum().item()
                total_ratio_count += ratio.numel()

            n_prompts += 1

        n = max(n_prompts, 1)
        return {
            "policy_loss": total_policy_loss / n,
            "kl": total_kl / n,
            "reward": total_reward / n,
            "clip_fraction": clip_count / max(total_ratio_count, 1),
        }

    @torch.no_grad()
    def evaluate(self, data):
        self.model.eval()
        from rewards import compute_f1, compute_exact_match

        f1s, ems, lengths = [], [], []
        for sample in data:
            prompt = self.build_prompt(sample["question"], sample["context"])
            enc = self.tokenizer(
                prompt, return_tensors="pt", truncation=True,
                max_length=config.MAX_INPUT_LENGTH,
            ).to(config.DEVICE)

            out = self.model.generate(
                **enc, max_new_tokens=config.MAX_OUTPUT_LENGTH,
                do_sample=False,
            )
            pred = self.tokenizer.decode(out[0], skip_special_tokens=True)
            gt = sample["answer"]

            f1s.append(compute_f1(pred, gt))
            ems.append(compute_exact_match(pred, gt))
            lengths.append(len(pred.split()))

        self.model.train()
        return {
            "f1": sum(f1s) / max(len(f1s), 1),
            "em": sum(ems) / max(len(ems), 1),
            "avg_length": sum(lengths) / max(len(lengths), 1),
        }

    def train(self, train_data, eval_data, epochs=config.GRPO_EPOCHS):
        import random

        total_steps = (len(train_data) // config.GRPO_BATCH_SIZE) * epochs
        warmup_steps = int(total_steps * config.GRPO_WARMUP_RATIO)
        scheduler = CosineAnnealingLR(self.optimizer, T_max=max(total_steps - warmup_steps, 1))

        history = {
            "rewards": [], "policy_losses": [], "kl_values": [],
            "clip_fractions": [], "eval_f1": [], "eval_em": [],
            "eval_steps": [],
        }

        best_f1 = -1.0
        patience_counter = 0
        global_step = 0

        print(f"\nGRPO Training: {epochs} epochs, {len(train_data)} samples, "
              f"batch={config.GRPO_BATCH_SIZE}, K={config.GRPO_GROUP_SIZE}")
        print(f"Total steps: ~{total_steps}, eval every {config.EVAL_EVERY_STEPS} steps")

        for epoch in range(1, epochs + 1):
            rng = random.Random(config.DATA_SEED + epoch)
            shuffled = train_data.copy()
            rng.shuffle(shuffled)

            n_batches = len(shuffled) // config.GRPO_BATCH_SIZE
            pbar = tqdm(range(n_batches), desc=f"GRPO Epoch {epoch}/{epochs}")

            for batch_idx in pbar:
                start = batch_idx * config.GRPO_BATCH_SIZE
                batch = shuffled[start : start + config.GRPO_BATCH_SIZE]

                metrics = self.train_step(batch)

                global_step += 1
                if global_step > warmup_steps:
                    scheduler.step()

                history["rewards"].append(metrics["reward"])
                history["policy_losses"].append(metrics["policy_loss"])
                history["kl_values"].append(metrics["kl"])
                history["clip_fractions"].append(metrics["clip_fraction"])

                pbar.set_postfix(
                    r=f"{metrics['reward']:.3f}",
                    kl=f"{metrics['kl']:.4f}",
                    clip=f"{metrics['clip_fraction']:.2f}",
                )

                # евалуација на секои EVAL_EVERY_STEPS чекори
                if global_step % config.EVAL_EVERY_STEPS == 0:
                    eval_metrics = self.evaluate(eval_data)
                    history["eval_f1"].append(eval_metrics["f1"])
                    history["eval_em"].append(eval_metrics["em"])
                    history["eval_steps"].append(global_step)

                    print(
                        f"\n  [Step {global_step}] Eval F1={eval_metrics['f1']:.4f}  "
                        f"EM={eval_metrics['em']:.4f}  "
                        f"avg_len={eval_metrics['avg_length']:.1f}"
                    )

                    if eval_metrics["f1"] > best_f1:
                        best_f1 = eval_metrics["f1"]
                        patience_counter = 0
                        self.save(config.GRPO_BEST_DIR)
                        print(f"  -> New best F1={best_f1:.4f}, saved to {config.GRPO_BEST_DIR}")
                    else:
                        patience_counter += 1
                        print(f"  -> No improvement ({patience_counter}/{config.EARLY_STOPPING_PATIENCE})")

                    if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                        print(f"\nEarly stopping at step {global_step} (patience exhausted).")
                        self.save(config.GRPO_FINAL_DIR)
                        return history

            eval_metrics = self.evaluate(eval_data)
            print(
                f"Epoch {epoch} done. Eval F1={eval_metrics['f1']:.4f}  "
                f"EM={eval_metrics['em']:.4f}"
            )

        self.save(config.GRPO_FINAL_DIR)
        print(f"GRPO training complete. Final model saved to {config.GRPO_FINAL_DIR}")
        return history

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path):
        from transformers import AutoModelForSeq2SeqLM
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path).to(config.DEVICE)
