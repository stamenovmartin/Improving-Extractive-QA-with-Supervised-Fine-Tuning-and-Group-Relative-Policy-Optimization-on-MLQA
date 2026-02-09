import os
import sys
import json
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from data import load_mlqa_test, load_mlqa_test_lang
from rewards import compute_f1, compute_exact_match
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm


def evaluate_model(model, tokenizer, test_data, label):
    model.eval()
    f1s, ems, lengths = [], [], []
    samples = []

    for item in tqdm(test_data, desc=f"Eval [{label}]"):
        prompt = (
            f"Answer the question based on the context.\n\n"
            f"Context: {item['context']}\n\nQuestion: {item['question']}"
        )
        enc = tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=config.MAX_INPUT_LENGTH,
        ).to(config.DEVICE)

        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=config.MAX_OUTPUT_LENGTH, do_sample=False)
        pred = tokenizer.decode(out[0], skip_special_tokens=True)
        gt = item["answer"]

        f1 = compute_f1(pred, gt)
        em = compute_exact_match(pred, gt)
        f1s.append(f1)
        ems.append(em)
        lengths.append(len(pred.split()))

        if len(samples) < 10:
            samples.append({
                "question": item["question"],
                "ground_truth": gt,
                "prediction": pred,
                "f1": f1,
                "exact_match": em,
            })

    return {
        "label": label,
        "metrics": {
            "avg_f1": float(np.mean(f1s)),
            "avg_exact_match": float(np.mean(ems)),
            "avg_length": float(np.mean(lengths)),
            "total_samples": len(f1s),
        },
        "samples": samples,
    }


def evaluate_translated_language(model, tokenizer, en_data, target_lang="mk",
                                 n_samples=100, label=None):
    from translate import get_translator

    if label is None:
        label = f"GRPO-best [{target_lang}-{target_lang}]"

    rng = random.Random(config.DATA_SEED)
    subset = rng.sample(en_data, min(n_samples, len(en_data)))

    print(f"\n--- Translated Zero-Shot: {target_lang}-{target_lang} ({len(subset)} samples) ---")
    translator = get_translator(target_lang)
    translated_data = translator.translate_dataset(subset)

    del translator
    torch.cuda.empty_cache() if config.DEVICE == "cuda" else None

    return evaluate_model(model, tokenizer, translated_data, label)


def plot_comparison(results, save_dir=config.PLOTS_DIR):
    os.makedirs(save_dir, exist_ok=True)
    labels = [r["label"] for r in results]
    f1s = [r["metrics"]["avg_f1"] for r in results]
    ems = [r["metrics"]["avg_exact_match"] for r in results]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, f1s, width, label="F1", color="#3498db")
    bars2 = ax.bar(x + width / 2, ems, width, label="Exact Match", color="#2ecc71")

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison: F1 and Exact Match on MLQA Test (en-en)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, max(max(f1s), max(ems)) * 1.3 + 0.05)
    ax.grid(True, alpha=0.3, axis="y")

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(save_dir, "evaluation_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot saved to {path}")


def main():
    n_samples = config.EVAL_TEST_SAMPLES
    for i, arg in enumerate(sys.argv):
        if arg == "--samples" and i + 1 < len(sys.argv):
            n_samples = int(sys.argv[i + 1])

    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    print("Loading MLQA test data...")
    test_data = load_mlqa_test()
    rng = random.Random(config.DATA_SEED)
    if len(test_data) > n_samples:
        test_data = rng.sample(test_data, n_samples)
    print(f"Test samples: {len(test_data)}")

    results = []
    tokenizer = AutoTokenizer.from_pretrained(config.GENERATOR_MODEL)

    print("\n--- Evaluating Base Model ---")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(config.GENERATOR_MODEL).to(config.DEVICE)
    results.append(evaluate_model(base_model, tokenizer, test_data, "Base"))
    del base_model
    torch.cuda.empty_cache() if config.DEVICE == "cuda" else None

    if os.path.exists(config.SFT_MODEL_DIR):
        print("\n--- Evaluating SFT Model ---")
        sft_model = AutoModelForSeq2SeqLM.from_pretrained(config.SFT_MODEL_DIR).to(config.DEVICE)
        sft_tokenizer = AutoTokenizer.from_pretrained(config.SFT_MODEL_DIR)
        results.append(evaluate_model(sft_model, sft_tokenizer, test_data, "SFT"))
        del sft_model
        torch.cuda.empty_cache() if config.DEVICE == "cuda" else None
    else:
        print(f"\nSFT model not found at {config.SFT_MODEL_DIR}, skipping.")

    if os.path.exists(config.GRPO_BEST_DIR):
        print("\n--- Evaluating GRPO Best Model ---")
        grpo_model = AutoModelForSeq2SeqLM.from_pretrained(config.GRPO_BEST_DIR).to(config.DEVICE)
        grpo_tokenizer = AutoTokenizer.from_pretrained(config.GRPO_BEST_DIR)
        results.append(evaluate_model(grpo_model, grpo_tokenizer, test_data, "GRPO-best"))
        del grpo_model
        torch.cuda.empty_cache() if config.DEVICE == "cuda" else None
    else:
        print(f"\nGRPO best model not found at {config.GRPO_BEST_DIR}, skipping.")

    if os.path.exists(config.GRPO_FINAL_DIR):
        print("\n--- Evaluating GRPO Final Model ---")
        grpo_f = AutoModelForSeq2SeqLM.from_pretrained(config.GRPO_FINAL_DIR).to(config.DEVICE)
        grpo_ft = AutoTokenizer.from_pretrained(config.GRPO_FINAL_DIR)
        results.append(evaluate_model(grpo_f, grpo_ft, test_data, "GRPO-final"))
        del grpo_f
        torch.cuda.empty_cache() if config.DEVICE == "cuda" else None
    else:
        print(f"\nGRPO final model not found at {config.GRPO_FINAL_DIR}, skipping.")

    # мултилингвална евалуација (zero-shot)
    MULTILINGUAL_PAIRS = [("de", "de"), ("es", "es")]
    MULTILINGUAL_SAMPLES = 100
    multilingual_results = []

    if os.path.exists(config.GRPO_BEST_DIR):
        print("\n--- Multilingual Zero-Shot Evaluation (GRPO-best) ---")
        ml_model = AutoModelForSeq2SeqLM.from_pretrained(config.GRPO_BEST_DIR).to(config.DEVICE)
        ml_tokenizer = AutoTokenizer.from_pretrained(config.GRPO_BEST_DIR)

        for ctx_lang, q_lang in MULTILINGUAL_PAIRS:
            try:
                ml_data = load_mlqa_test_lang(ctx_lang, q_lang)
                ml_rng = random.Random(config.DATA_SEED)
                if len(ml_data) > MULTILINGUAL_SAMPLES:
                    ml_data = ml_rng.sample(ml_data, MULTILINGUAL_SAMPLES)
                label = f"GRPO-best [{ctx_lang}-{q_lang}]"
                ml_result = evaluate_model(ml_model, ml_tokenizer, ml_data, label)
                multilingual_results.append(ml_result)
            except FileNotFoundError as e:
                print(f"  Skipping {ctx_lang}-{q_lang}: {e}")

        mk_result = evaluate_translated_language(
            ml_model, ml_tokenizer, test_data,
            target_lang="mk", n_samples=MULTILINGUAL_SAMPLES,
        )
        multilingual_results.append(mk_result)

        del ml_model
        torch.cuda.empty_cache() if config.DEVICE == "cuda" else None
    else:
        print("\nGRPO-best not found, skipping multilingual evaluation.")

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS (English-English, MLQA test split)")
    print("=" * 70)
    for r in results:
        m = r["metrics"]
        print(f"  {r['label']:15s}  F1={m['avg_f1']:.4f}  EM={m['avg_exact_match']:.4f}  "
              f"avg_len={m['avg_length']:.1f}  (n={m['total_samples']})")

    if multilingual_results:
        print("\n" + "-" * 70)
        print("ZERO-SHOT MULTILINGUAL (GRPO-best, English-trained)")
        print("-" * 70)
        for r in multilingual_results:
            m = r["metrics"]
            print(f"  {r['label']:25s}  F1={m['avg_f1']:.4f}  EM={m['avg_exact_match']:.4f}  "
                  f"avg_len={m['avg_length']:.1f}  (n={m['total_samples']})")

    print("\nNote: GRPO-best checkpoint is reported alongside GRPO-final due to")
    print("non-monotonic RL validation behaviour (best F1 at step 200, declining after).")
    print("=" * 70)

    plot_comparison(results)

    all_results_dict = {}
    for r in results:
        all_results_dict[r["label"]] = {"metrics": r["metrics"], "samples": r["samples"]}
    for r in multilingual_results:
        all_results_dict[r["label"]] = {"metrics": r["metrics"], "samples": r["samples"]}

    out_path = os.path.join(config.LOGS_DIR, "evaluation_results.json")
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results_dict, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
