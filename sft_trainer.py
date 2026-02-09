import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os

import config


class QADataset(Dataset):
    def __init__(self, qa_pairs, tokenizer, max_input_len=config.MAX_INPUT_LENGTH,
                 max_target_len=config.MAX_OUTPUT_LENGTH):
        self.qa_pairs = qa_pairs
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, idx):
        question, context, answer = self.qa_pairs[idx]
        input_text = (
            f"Answer the question based on the context.\n\n"
            f"Context: {context}\n\nQuestion: {question}"
        )

        inputs = self.tokenizer(
            input_text, truncation=True, max_length=self.max_input_len,
            padding="max_length", return_tensors="pt"
        )
        targets = self.tokenizer(
            answer, truncation=True, max_length=self.max_target_len,
            padding="max_length", return_tensors="pt"
        )

        labels = targets["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels,
        }


class SFTTrainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.SFT_LR)
        self.scaler = torch.amp.GradScaler(enabled=config.USE_AMP)

    def train(self, train_pairs, eval_pairs, epochs=config.SFT_EPOCHS):
        train_ds = QADataset(train_pairs, self.tokenizer)
        eval_ds = QADataset(eval_pairs, self.tokenizer)
        train_loader = DataLoader(train_ds, batch_size=config.SFT_BATCH_SIZE, shuffle=True)
        eval_loader = DataLoader(eval_ds, batch_size=config.SFT_BATCH_SIZE)

        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * config.SFT_WARMUP_RATIO)
        scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps - warmup_steps)

        history = {"train_loss": [], "eval_loss": [], "batch_losses": []}
        best_eval_loss = float("inf")
        global_step = 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss, n_batches = 0.0, 0
            pbar = tqdm(train_loader, desc=f"SFT Epoch {epoch}/{epochs}")

            for batch in pbar:
                input_ids = batch["input_ids"].to(config.DEVICE)
                attention_mask = batch["attention_mask"].to(config.DEVICE)
                labels = batch["labels"].to(config.DEVICE)

                with torch.amp.autocast(device_type=config.DEVICE, enabled=config.USE_AMP):
                    loss = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    ).loss

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), config.SFT_MAX_GRAD_NORM
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()

                global_step += 1
                if global_step > warmup_steps:
                    scheduler.step()

                lv = loss.item()
                total_loss += lv
                n_batches += 1
                history["batch_losses"].append(lv)
                pbar.set_postfix(loss=f"{lv:.4f}")

            avg_train = total_loss / max(n_batches, 1)
            history["train_loss"].append(avg_train)

            eval_loss = self._evaluate(eval_loader)
            history["eval_loss"].append(eval_loss)

            print(f"Epoch {epoch}: train_loss={avg_train:.4f}  eval_loss={eval_loss:.4f}")

            if eval_loss < best_eval_loss or epoch == 1:
                best_eval_loss = eval_loss
                self.save(config.SFT_MODEL_DIR)
                print(f"  -> Best eval loss ({eval_loss:.4f}), saved to {config.SFT_MODEL_DIR}")

        return history

    @torch.no_grad()
    def _evaluate(self, loader):
        self.model.eval()
        total, n = 0.0, 0
        for batch in loader:
            input_ids = batch["input_ids"].to(config.DEVICE)
            attention_mask = batch["attention_mask"].to(config.DEVICE)
            labels = batch["labels"].to(config.DEVICE)
            with torch.amp.autocast(device_type=config.DEVICE, enabled=config.USE_AMP):
                loss = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                ).loss
            total += loss.item()
            n += 1
        self.model.train()
        return total / max(n, 1)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path):
        from transformers import AutoModelForSeq2SeqLM
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path).to(config.DEVICE)
