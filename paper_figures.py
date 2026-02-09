# Генерирање слики и табели за трудот.
# python paper_figures.py              -> од постоечки JSON (без GPU)
# python paper_figures.py --full-eval  -> ре-евалуација на моделите

import os
import sys
import json
import random
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import config
from rewards import compute_f1, compute_exact_match

PLOTS_DIR = config.PLOTS_DIR
LOGS_DIR = config.LOGS_DIR
TABLES_DIR = os.path.join(config.BASE_DIR, "tables")

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.figsize": (8, 5),
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def load_eval_results():
    path = os.path.join(LOGS_DIR, "evaluation_results.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_sft_history():
    path = os.path.join(LOGS_DIR, "sft_history.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_grpo_history():
    path = os.path.join(LOGS_DIR, "grpo_history.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dirs():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)


def save_fig(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def fig_crosslingual(results):
    """Споредба на F1 и EM по јазици."""
    langs = [
        ("en-en", "GRPO-best", "English"),
        ("es-es", "GRPO-best [es-es]", "Spanish"),
        ("de-de", "GRPO-best [de-de]", "German"),
        ("mk-mk", "GRPO-best [mk-mk]", "Macedonian"),
    ]

    lang_colors = {"en": "#3498db", "es": "#e74c3c", "de": "#2ecc71", "mk": "#9b59b6"}

    labels, f1s, ems, colors = [], [], [], []
    for code, key, label in langs:
        if key not in results:
            continue
        m = results[key]["metrics"]
        labels.append(label)
        f1s.append(m["avg_f1"])
        ems.append(m["avg_exact_match"])
        colors.append(lang_colors.get(code.split("-")[0], "#666"))

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars_f1 = ax.bar(x - w / 2, f1s, w, label="F1", color=colors, alpha=0.85)
    bars_em = ax.bar(x + w / 2, ems, w, label="Exact Match", color=colors, alpha=0.50)

    for bars in [bars_f1, bars_em]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Score")
    ax.set_title("Zero-Shot Cross-Lingual Transfer (GRPO-best, trained on English)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, max(f1s) * 1.25 + 0.05)

    save_fig(fig, "crosslingual_comparison.png")


def fig_training_pipeline(sft_hist, grpo_hist):
    """2x2 панел: SFT loss, GRPO reward, KL, eval метрики."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Training Pipeline Overview", fontsize=14, fontweight="bold")

    # SFT train vs eval loss
    ax = axes[0, 0]
    epochs = list(range(1, len(sft_hist["train_loss"]) + 1))
    ax.plot(epochs, sft_hist["train_loss"], "o-", color="#3498db", label="Train loss", linewidth=2)
    ax.plot(epochs, sft_hist["eval_loss"], "s-", color="#e74c3c", label="Eval loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Phase 1: SFT Loss")
    ax.legend()
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # GRPO reward (smoothed)
    ax = axes[0, 1]
    rewards = grpo_hist["rewards"]
    steps = list(range(1, len(rewards) + 1))
    window = 50
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
    ax.plot(steps, rewards, alpha=0.25, color="#3498db")
    ax.plot(steps[window - 1:], smoothed, color="#e74c3c", linewidth=2, label=f"Smoothed (w={window})")
    ax.set_xlabel("Step")
    ax.set_ylabel("Avg Reward")
    ax.set_title("Phase 2: GRPO Reward")
    ax.legend()

    # KL divergence
    ax = axes[1, 0]
    kl_key = "kl_values" if "kl_values" in grpo_hist else "kl"
    if kl_key in grpo_hist:
        kl = grpo_hist[kl_key]
        kl_steps = list(range(1, len(kl) + 1))
        ax.plot(kl_steps, kl, alpha=0.3, color="#9b59b6")
        kl_smooth = np.convolve(kl, np.ones(window) / window, mode="valid")
        ax.plot(kl_steps[window - 1:], kl_smooth, color="#8e44ad", linewidth=2,
                label=f"Smoothed (w={window})")
        ax.set_xlabel("Step")
        ax.set_ylabel("KL Divergence")
        ax.set_title("Phase 2: KL(policy || reference)")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "KL data not available", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="gray")
        ax.set_title("Phase 2: KL Divergence")

    # Eval F1/EM during GRPO
    ax = axes[1, 1]
    eval_steps = grpo_hist.get("eval_steps", [])
    eval_f1 = grpo_hist.get("eval_f1", [])
    eval_em = grpo_hist.get("eval_em", [])
    if eval_steps and eval_f1:
        ax.plot(eval_steps, eval_f1, "o-", color="#3498db", linewidth=2,
                markersize=8, label="F1")
        ax.plot(eval_steps, eval_em, "s-", color="#e74c3c", linewidth=2,
                markersize=8, label="EM")
        for i, (s, f, e) in enumerate(zip(eval_steps, eval_f1, eval_em)):
            ax.annotate(f"{f:.3f}", (s, f), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=9)
        ax.set_xlabel("Step")
        ax.set_ylabel("Score")
        ax.legend()
    ax.set_title("Phase 2: Eval Metrics During GRPO")

    plt.tight_layout()
    save_fig(fig, "training_pipeline.png")


def fig_grpo_detailed(grpo_hist):
    """3-panel GRPO diagnostics: KL, clip fraction, eval curve."""
    has_kl = "kl_values" in grpo_hist
    has_clip = "clip_fractions" in grpo_hist
    has_eval = "eval_steps" in grpo_hist and "eval_f1" in grpo_hist

    n_panels = sum([has_kl, has_clip, has_eval])
    if n_panels == 0:
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]
    fig.suptitle("GRPO Training Diagnostics", fontsize=14, fontweight="bold")

    window = 50
    idx = 0

    if has_kl:
        ax = axes[idx]; idx += 1
        kl = grpo_hist["kl_values"]
        steps = list(range(1, len(kl) + 1))
        ax.plot(steps, kl, alpha=0.25, color="#9b59b6")
        kl_smooth = np.convolve(kl, np.ones(window) / window, mode="valid")
        ax.plot(steps[window - 1:], kl_smooth, color="#8e44ad", linewidth=2,
                label=f"Smoothed (w={window})")
        ax.set_xlabel("Step")
        ax.set_ylabel("KL Divergence")
        ax.set_title("KL(policy || reference)")
        ax.legend()

    if has_clip:
        ax = axes[idx]; idx += 1
        clip = grpo_hist["clip_fractions"]
        steps = list(range(1, len(clip) + 1))
        ax.plot(steps, clip, alpha=0.25, color="#e67e22")
        clip_smooth = np.convolve(clip, np.ones(window) / window, mode="valid")
        ax.plot(steps[window - 1:], clip_smooth, color="#d35400", linewidth=2,
                label=f"Smoothed (w={window})")
        ax.set_xlabel("Step")
        ax.set_ylabel("Clip Fraction")
        ax.set_title("PPO Clip Fraction")
        ax.set_ylim(-0.05, 1.05)
        ax.legend()

    if has_eval:
        ax = axes[idx]; idx += 1
        eval_steps = grpo_hist["eval_steps"]
        eval_f1 = grpo_hist["eval_f1"]
        eval_em = grpo_hist["eval_em"]
        ax.plot(eval_steps, eval_f1, "o-", color="#3498db", linewidth=2,
                markersize=8, label="F1")
        ax.plot(eval_steps, eval_em, "s-", color="#e74c3c", linewidth=2,
                markersize=8, label="EM")
        for s, f in zip(eval_steps, eval_f1):
            ax.annotate(f"{f:.3f}", (s, f), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=9)
        ax.set_xlabel("Step")
        ax.set_ylabel("Score")
        ax.set_title("Eval Metrics During Training")
        ax.legend()

    plt.tight_layout()
    save_fig(fig, "grpo_diagnostics.png")


def fig_model_comparison(results):
    """Bar chart with delta annotations for model comparison."""
    models = ["Base", "SFT", "GRPO-best", "GRPO-final"]
    available = [m for m in models if m in results]

    f1s = [results[m]["metrics"]["avg_f1"] for m in available]
    ems = [results[m]["metrics"]["avg_exact_match"] for m in available]
    base_f1 = f1s[0]

    x = np.arange(len(available))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - w / 2, f1s, w, label="F1", color="#3498db", alpha=0.85)
    bars2 = ax.bar(x + w / 2, ems, w, label="Exact Match", color="#2ecc71", alpha=0.85)

    for i, bar in enumerate(bars1):
        h = bar.get_height()
        delta = ((h - base_f1) / base_f1) * 100 if i > 0 else 0
        label = f"{h:.3f}"
        if i > 0:
            label += f"\n(+{delta:.1f}%)"
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                label, ha="center", va="bottom", fontsize=9, fontweight="bold")

    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison: F1 and Exact Match on MLQA Test (en-en, n=500)")
    ax.set_xticks(x)
    ax.set_xticklabels(available)
    ax.legend()
    ax.set_ylim(0, max(max(f1s), max(ems)) * 1.25 + 0.05)

    save_fig(fig, "model_comparison.png")


def fig_reward_distribution(grpo_hist):
    """Violin plot of rewards across training step ranges."""
    rewards = grpo_hist["rewards"]
    n = len(rewards)
    chunk_size = n // 4
    chunks = []
    labels = []
    for i in range(4):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < 3 else n
        chunks.append(rewards[start:end])
        labels.append(f"Steps {start + 1}-{end}")

    fig, ax = plt.subplots(figsize=(9, 5))
    parts = ax.violinplot(chunks, showmeans=True, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("#3498db")
        pc.set_alpha(0.6)
    parts["cmeans"].set_color("#e74c3c")
    parts["cmedians"].set_color("#2ecc71")

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Reward")
    ax.set_title("GRPO Reward Distribution Across Training")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#e74c3c", linewidth=2, label="Mean"),
        Line2D([0], [0], color="#2ecc71", linewidth=2, label="Median"),
    ]
    ax.legend(handles=legend_elements)

    save_fig(fig, "reward_distribution.png")


def fig_f1_boxplot(per_sample_data):
    """Box plot of per-sample F1 scores across models."""
    models = ["Base", "SFT", "GRPO-best", "GRPO-final"]
    available = [m for m in models if m in per_sample_data]
    data = [per_sample_data[m]["f1_scores"] for m in available]

    fig, ax = plt.subplots(figsize=(9, 6))
    bp = ax.boxplot(data, labels=available, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2),
                    flierprops=dict(marker=".", markersize=3, alpha=0.4))

    colors = ["#95a5a6", "#3498db", "#2ecc71", "#e67e22"]
    for patch, color in zip(bp["boxes"], colors[:len(available)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    means = [np.mean(d) for d in data]
    ax.scatter(range(1, len(available) + 1), means, marker="D", color="#e74c3c",
               s=60, zorder=5, label="Mean")
    for i, m in enumerate(means):
        ax.annotate(f"{m:.3f}", (i + 1, m), textcoords="offset points",
                    xytext=(25, 0), fontsize=9, color="#e74c3c")

    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Sample F1 Score Distribution (n=500)")
    ax.legend()

    save_fig(fig, "f1_boxplot.png")


def categorize_errors(f1_scores, em_scores):
    exact, partial, wrong = 0, 0, 0
    for f1, em in zip(f1_scores, em_scores):
        if em == 1.0:
            exact += 1
        elif f1 > 0.0:
            partial += 1
        else:
            wrong += 1
    total = len(f1_scores)
    return {
        "Exact Match": exact / total,
        "Partial Match (F1>0)": partial / total,
        "No Overlap (F1=0)": wrong / total,
    }


def fig_error_analysis(per_sample_data):
    """Stacked bar chart of error categories per model."""
    models = ["Base", "SFT", "GRPO-best", "GRPO-final"]
    available = [m for m in models if m in per_sample_data]

    categories_per_model = {}
    for m in available:
        categories_per_model[m] = categorize_errors(
            per_sample_data[m]["f1_scores"],
            per_sample_data[m]["em_scores"],
        )

    cat_names = ["Exact Match", "Partial Match (F1>0)", "No Overlap (F1=0)"]
    cat_colors = ["#2ecc71", "#f1c40f", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(available))
    w = 0.5
    bottom = np.zeros(len(available))

    for cat, color in zip(cat_names, cat_colors):
        vals = [categories_per_model[m][cat] for m in available]
        ax.bar(x, vals, w, bottom=bottom, label=cat, color=color, alpha=0.85)
        for i, (v, b) in enumerate(zip(vals, bottom)):
            if v > 0.05:
                ax.text(i, b + v / 2, f"{v:.1%}", ha="center", va="center",
                        fontsize=9, fontweight="bold")
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(available)
    ax.set_ylabel("Proportion")
    ax.set_title("Error Analysis: Prediction Categories by Model")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    save_fig(fig, "error_analysis.png")


def table_main_results(results):
    """LaTeX table for main English results."""
    models = [
        ("Base", "Base (flan-t5-small)"),
        ("SFT", "+ SFT"),
        ("GRPO-best", "+ GRPO (best, step 200)"),
        ("GRPO-final", "+ GRPO (final, step 774)"),
    ]
    base_f1 = results["Base"]["metrics"]["avg_f1"]

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Model performance on MLQA English-English test set (n=500).}")
    lines.append(r"\label{tab:main_results}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Model & F1 & EM & Avg Length & $\Delta$ F1 \\")
    lines.append(r"\midrule")

    for key, label in models:
        if key not in results:
            continue
        m = results[key]["metrics"]
        f1 = m["avg_f1"]
        em = m["avg_exact_match"]
        length = m["avg_length"]
        delta = ((f1 - base_f1) / base_f1) * 100

        bold_f1 = f"\\textbf{{{f1:.4f}}}" if key == "GRPO-best" else f"{f1:.4f}"
        bold_em = f"\\textbf{{{em:.3f}}}" if key == "GRPO-best" else f"{em:.3f}"
        delta_str = "---" if key == "Base" else f"+{delta:.1f}\\%"

        lines.append(f"{label} & {bold_f1} & {bold_em} & {length:.2f} & {delta_str} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    text = "\n".join(lines)
    path = os.path.join(TABLES_DIR, "main_results.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  Saved: {path}")
    return text


def table_crosslingual(results):
    langs = [
        ("GRPO-best", "English (en-en)", 500),
        ("GRPO-best [es-es]", "Spanish (es-es)", 100),
        ("GRPO-best [de-de]", "German (de-de)", 100),
        ("GRPO-best [mk-mk]", "Macedonian (mk-mk)", 100),
    ]

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Zero-shot cross-lingual transfer of GRPO-best model.}")
    lines.append(r"\label{tab:crosslingual}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Language & F1 & EM & n \\")
    lines.append(r"\midrule")

    for key, label, n in langs:
        if key not in results:
            continue
        m = results[key]["metrics"]
        lines.append(f"{label} & {m['avg_f1']:.4f} & {m['avg_exact_match']:.3f} & {n} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    text = "\n".join(lines)
    path = os.path.join(TABLES_DIR, "crosslingual.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  Saved: {path}")
    return text


def table_hyperparameters():
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Hyperparameters for SFT and GRPO training phases.}")
    lines.append(r"\label{tab:hyperparameters}")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(r"Hyperparameter & SFT & GRPO \\")
    lines.append(r"\midrule")
    lines.append(f"Learning rate & {config.SFT_LR} & {config.GRPO_LR} \\\\")
    lines.append(f"Batch size & {config.SFT_BATCH_SIZE} & {config.GRPO_BATCH_SIZE} \\\\")
    lines.append(f"Epochs & {config.SFT_EPOCHS} & {config.GRPO_EPOCHS} \\\\")
    lines.append(f"Max grad norm & {config.SFT_MAX_GRAD_NORM} & {config.GRPO_MAX_GRAD_NORM} \\\\")
    lines.append(f"Warmup ratio & {config.SFT_WARMUP_RATIO} & {config.GRPO_WARMUP_RATIO} \\\\")
    lines.append(r"\midrule")
    lines.append(f"Group size (K) & --- & {config.GRPO_GROUP_SIZE} \\\\")
    lines.append(f"Clip $\\epsilon$ & --- & {config.GRPO_CLIP_EPS} \\\\")
    lines.append(f"KL coefficient $\\beta$ & --- & {config.GRPO_KL_BETA} \\\\")
    lines.append(f"Temperature & --- & {config.GRPO_TEMPERATURE} \\\\")
    lines.append(f"Top-p & --- & {config.GRPO_TOP_P} \\\\")
    lines.append(r"\midrule")
    lines.append(f"Reward: F1 weight & --- & {config.REWARD_F1_WEIGHT} \\\\")
    lines.append(f"Reward: EM weight & --- & {config.REWARD_EM_WEIGHT} \\\\")
    lines.append(f"Length penalty & --- & {config.REWARD_LENGTH_PENALTY} \\\\")
    lines.append(r"\midrule")
    lines.append(f"Max input length & \\multicolumn{{2}}{{c}}{{{config.MAX_INPUT_LENGTH}}} \\\\")
    lines.append(f"Max output length & \\multicolumn{{2}}{{c}}{{{config.MAX_OUTPUT_LENGTH}}} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    text = "\n".join(lines)
    path = os.path.join(TABLES_DIR, "hyperparameters.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  Saved: {path}")
    return text


def table_qualitative_examples(results):
    """Side-by-side predictions from different models."""
    models = ["Base", "SFT", "GRPO-best", "GRPO-final"]
    available = [m for m in models if m in results]

    ref_samples = results[available[0]]["samples"]
    n_examples = min(5, len(ref_samples))

    indices = []
    correct, partial, wrong = [], [], []
    for i, s in enumerate(ref_samples):
        if s["exact_match"] == 1.0:
            correct.append(i)
        elif s["f1"] > 0:
            partial.append(i)
        else:
            wrong.append(i)

    indices = correct[:2] + partial[:2] + wrong[:1]
    if len(indices) < n_examples:
        for i in range(len(ref_samples)):
            if i not in indices:
                indices.append(i)
            if len(indices) >= n_examples:
                break

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Qualitative comparison of model predictions on MLQA test samples.}")
    lines.append(r"\label{tab:qualitative}")
    cols = "p{4cm}" + "p{2.5cm}" * (len(available) + 1)
    lines.append(r"\begin{tabular}{" + cols + "}")
    lines.append(r"\toprule")
    header = "Question & Ground Truth & " + " & ".join(available) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for idx in indices:
        q = ref_samples[idx]["question"]
        gt = ref_samples[idx]["ground_truth"]
        q = q.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
        gt = gt.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")

        preds = []
        for m in available:
            if idx < len(results[m]["samples"]):
                pred = results[m]["samples"][idx]["prediction"]
                f1 = results[m]["samples"][idx]["f1"]
                pred = pred.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
                preds.append(f"{pred} \\tiny{{({f1:.2f})}}")
            else:
                preds.append("---")

        row = f"{q} & {gt} & " + " & ".join(preds) + r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    text = "\n".join(lines)
    path = os.path.join(TABLES_DIR, "qualitative.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  Saved: {path}")
    return text


def table_error_analysis(per_sample_data):
    models = ["Base", "SFT", "GRPO-best", "GRPO-final"]
    available = [m for m in models if m in per_sample_data]

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Error analysis: prediction categories by model (n=500).}")
    lines.append(r"\label{tab:error_analysis}")
    lines.append(r"\begin{tabular}{l" + "c" * len(available) + "}")
    lines.append(r"\toprule")
    lines.append("Category & " + " & ".join(available) + r" \\")
    lines.append(r"\midrule")

    cats = {}
    for m in available:
        cats[m] = categorize_errors(
            per_sample_data[m]["f1_scores"],
            per_sample_data[m]["em_scores"],
        )

    for cat in ["Exact Match", "Partial Match (F1>0)", "No Overlap (F1=0)"]:
        vals = [f"{cats[m][cat]:.1%}" for m in available]
        lines.append(f"{cat} & " + " & ".join(vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    text = "\n".join(lines)
    path = os.path.join(TABLES_DIR, "error_analysis.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  Saved: {path}")
    return text


def run_full_evaluation(n_samples=500):
    """Ре-евалуација на сите модели, зачувај per-sample F1/EM."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from data import load_mlqa_test
    from tqdm import tqdm

    print("\nLoading MLQA test data...")
    test_data = load_mlqa_test()
    rng = random.Random(config.DATA_SEED)
    if len(test_data) > n_samples:
        test_data = rng.sample(test_data, n_samples)
    print(f"Test samples: {len(test_data)}")

    model_paths = {
        "Base": config.GENERATOR_MODEL,
        "SFT": config.SFT_MODEL_DIR,
        "GRPO-best": config.GRPO_BEST_DIR,
        "GRPO-final": config.GRPO_FINAL_DIR,
    }

    per_sample_data = {}

    for label, path in model_paths.items():
        if not os.path.exists(path) and label != "Base":
            print(f"  Skipping {label}: {path} not found")
            continue

        print(f"\n  Evaluating {label}...")
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSeq2SeqLM.from_pretrained(path).to(config.DEVICE)
        model.eval()

        f1_scores, em_scores, predictions = [], [], []

        for item in tqdm(test_data, desc=f"  [{label}]"):
            prompt = (
                f"Answer the question based on the context.\n\n"
                f"Context: {item['context']}\n\nQuestion: {item['question']}"
            )
            enc = tokenizer(
                prompt, return_tensors="pt", truncation=True,
                max_length=config.MAX_INPUT_LENGTH,
            ).to(config.DEVICE)

            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=config.MAX_OUTPUT_LENGTH,
                                     do_sample=False)
            pred = tokenizer.decode(out[0], skip_special_tokens=True)
            gt = item["answer"]

            f1 = compute_f1(pred, gt)
            em = compute_exact_match(pred, gt)
            f1_scores.append(f1)
            em_scores.append(em)
            predictions.append({
                "question": item["question"],
                "ground_truth": gt,
                "prediction": pred,
                "f1": f1,
                "exact_match": em,
            })

        per_sample_data[label] = {
            "f1_scores": f1_scores,
            "em_scores": em_scores,
            "predictions": predictions,
            "avg_f1": float(np.mean(f1_scores)),
            "avg_em": float(np.mean(em_scores)),
        }

        del model
        if config.DEVICE == "cuda":
            torch.cuda.empty_cache()

    save_path = os.path.join(LOGS_DIR, "detailed_evaluation.json")
    serializable = {}
    for k, v in per_sample_data.items():
        serializable[k] = {
            "f1_scores": v["f1_scores"],
            "em_scores": v["em_scores"],
            "avg_f1": v["avg_f1"],
            "avg_em": v["avg_em"],
            "predictions": v["predictions"][:20],
        }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"\n  Detailed results saved to {save_path}")

    return per_sample_data


def main():
    full_eval = "--full-eval" in sys.argv
    ensure_dirs()

    print("Loading existing results...")
    results = load_eval_results()
    sft_hist = load_sft_history()
    grpo_hist = load_grpo_history()

    print("\nGenerating figures...")
    fig_crosslingual(results)
    fig_training_pipeline(sft_hist, grpo_hist)
    fig_grpo_detailed(grpo_hist)
    fig_model_comparison(results)
    fig_reward_distribution(grpo_hist)

    print("\nGenerating LaTeX tables...")
    print(table_main_results(results))
    print(table_crosslingual(results))
    print(table_hyperparameters())
    print(table_qualitative_examples(results))

    if full_eval:
        print("\nRunning full per-sample evaluation...")
        per_sample_data = run_full_evaluation()
        fig_f1_boxplot(per_sample_data)
        fig_error_analysis(per_sample_data)
        print(table_error_analysis(per_sample_data))
    else:
        detail_path = os.path.join(LOGS_DIR, "detailed_evaluation.json")
        if os.path.exists(detail_path):
            print(f"\nLoading cached detailed evaluation from {detail_path}")
            with open(detail_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            fig_f1_boxplot(cached)
            fig_error_analysis(cached)
            print(table_error_analysis(cached))
        else:
            print("\nSkipping box plot & error analysis (need --full-eval or cached data)")

    print("\nOutput files:")
    for root, _, files in os.walk(PLOTS_DIR):
        for f in sorted(files):
            if f.endswith(".png"):
                print(f"  [PLOT]  {os.path.join(root, f)}")
    for root, _, files in os.walk(TABLES_DIR):
        for f in sorted(files):
            if f.endswith(".tex"):
                print(f"  [TABLE] {os.path.join(root, f)}")


if __name__ == "__main__":
    main()
