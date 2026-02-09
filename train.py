import os
import sys
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from data import load_mlqa_validation, split_train_eval, as_tuples
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def plot_sft(history, save_dir=config.PLOTS_DIR):
    os.makedirs(save_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Phase 1: Supervised Fine-Tuning", fontsize=14, fontweight="bold")

    # Epoch-level losses
    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], "o-", label="Train loss")
    ax1.plot(epochs, history["eval_loss"], "s-", label="Eval loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Train vs Eval Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Batch-level loss
    ax2.plot(history["batch_losses"], alpha=0.4, color="blue")
    window = min(50, len(history["batch_losses"]) // 5) or 1
    if window > 1 and len(history["batch_losses"]) > window:
        import numpy as np
        smoothed = np.convolve(
            history["batch_losses"],
            np.ones(window) / window,
            mode="valid",
        )
        ax2.plot(range(window - 1, window - 1 + len(smoothed)), smoothed,
                 color="red", linewidth=2, label=f"Smoothed (w={window})")
        ax2.legend()
    ax2.set_xlabel("Batch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Batch Loss")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "sft_training.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SFT plot saved to {save_dir}/sft_training.png")


def plot_grpo(history, save_dir=config.PLOTS_DIR):
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Phase 2: GRPO Training", fontsize=14, fontweight="bold")

    steps = range(1, len(history["rewards"]) + 1)

    # Rewards
    ax = axes[0, 0]
    ax.plot(steps, history["rewards"], alpha=0.3, color="blue")
    w = min(50, len(history["rewards"]) // 5) or 1
    if w > 1 and len(history["rewards"]) > w:
        import numpy as np
        sm = np.convolve(history["rewards"], np.ones(w) / w, mode="valid")
        ax.plot(range(w - 1, w - 1 + len(sm)), sm, color="red", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.set_title("Average Reward per Step")
    ax.grid(True, alpha=0.3)

    # Policy loss
    ax = axes[0, 1]
    ax.plot(steps, history["policy_losses"], alpha=0.3, color="orange")
    if w > 1 and len(history["policy_losses"]) > w:
        import numpy as np
        sm = np.convolve(history["policy_losses"], np.ones(w) / w, mode="valid")
        ax.plot(range(w - 1, w - 1 + len(sm)), sm, color="darkred", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Policy Loss")
    ax.set_title("Policy Loss")
    ax.grid(True, alpha=0.3)

    # KL divergence
    ax = axes[1, 0]
    ax.plot(steps, history["kl_values"], alpha=0.3, color="purple")
    if w > 1 and len(history["kl_values"]) > w:
        import numpy as np
        sm = np.convolve(history["kl_values"], np.ones(w) / w, mode="valid")
        ax.plot(range(w - 1, w - 1 + len(sm)), sm, color="darkviolet", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("KL Divergence")
    ax.set_title("KL(policy || reference)")
    ax.grid(True, alpha=0.3)

    # Eval F1 / EM over training
    ax = axes[1, 1]
    if history["eval_steps"]:
        ax.plot(history["eval_steps"], history["eval_f1"], "o-", label="Eval F1")
        ax.plot(history["eval_steps"], history["eval_em"], "s-", label="Eval EM")
        ax.set_xlabel("Step")
        ax.set_ylabel("Score")
        ax.set_title("Eval Metrics During Training")
        ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "grpo_training.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"GRPO plot saved to {save_dir}/grpo_training.png")


def main():
    grpo_only = "--grpo-only" in sys.argv

    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)

    print("Loading MLQA validation data...")
    all_data = load_mlqa_validation()
    train_data, eval_data = split_train_eval(all_data)
    print(f"Train: {len(train_data)}, Eval: {len(eval_data)}")

    train_tuples = as_tuples(train_data)
    eval_tuples = as_tuples(eval_data)

    tokenizer = AutoTokenizer.from_pretrained(config.GENERATOR_MODEL)

    if not grpo_only:
        print("\n" + "=" * 60)
        print("PHASE 1: Supervised Fine-Tuning")
        print("=" * 60)

        model = AutoModelForSeq2SeqLM.from_pretrained(config.GENERATOR_MODEL).to(config.DEVICE)
        from sft_trainer import SFTTrainer
        sft = SFTTrainer(model, tokenizer)
        sft_history = sft.train(train_tuples, eval_tuples)
        plot_sft(sft_history)

        with open(os.path.join(config.LOGS_DIR, "sft_history.json"), "w") as f:
            json.dump(sft_history, f, indent=2)
        print("SFT phase complete.\n")
    else:
        print(f"\nSkipping SFT, loading model from {config.SFT_MODEL_DIR}")

    print("=" * 60)
    print("PHASE 2: GRPO (Group Relative Policy Optimization)")
    print("=" * 60)

    sft_path = config.SFT_MODEL_DIR
    if not os.path.exists(sft_path):
        print(f"ERROR: SFT model not found at {sft_path}. Run SFT first.")
        return
    model = AutoModelForSeq2SeqLM.from_pretrained(sft_path).to(config.DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(sft_path)

    from grpo_trainer import GRPOTrainer
    grpo = GRPOTrainer(model, tokenizer)
    grpo_history = grpo.train(train_data, eval_data)
    plot_grpo(grpo_history)

    with open(os.path.join(config.LOGS_DIR, "grpo_history.json"), "w") as f:
        json.dump(grpo_history, f, indent=2)

    print("\nTraining complete!")
    print(f"  SFT model:       {config.SFT_MODEL_DIR}")
    print(f"  GRPO best model: {config.GRPO_BEST_DIR}")
    print(f"  GRPO final model:{config.GRPO_FINAL_DIR}")
    print(f"  Plots:           {config.PLOTS_DIR}/")
    print(f"  Logs:            {config.LOGS_DIR}/")


if __name__ == "__main__":
    main()
