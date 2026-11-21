"""Chart helpers. All colours are set explicitly so figures render correctly on any background."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK = "#111827"
MUTED = "#6B7280"
GRID = "#E5E7EB"
PALETTE = {
    "zero-shot": "#9CA3AF",
    "tfidf": "#60A5FA",
    "fine-tuned": "#059669",
}


def _style(ax):
    ax.set_facecolor("white")
    ax.tick_params(colors=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_loss_curve(log_history, path, title="Qwen2.5-0.5B"):
    tr = [(e["step"], e["loss"]) for e in log_history if "loss" in e]
    ev = [(e["step"], e["eval_loss"]) for e in log_history if "eval_loss" in e]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    fig.patch.set_facecolor("white")
    _style(ax)
    ax.plot(*zip(*tr), label="Train loss", color="#2563EB", linewidth=2)
    if ev:
        ax.plot(*zip(*ev), label="Validation loss", color="#EA580C", linewidth=2, marker="o", markersize=4)
    ax.set_yscale("log")
    ax.set_xlabel("Optimizer step", color=INK)
    ax.set_ylabel("Cross-entropy loss (completion only, log scale)", color=INK)
    ax.set_title(f"QLoRA fine-tuning loss: {title}", color=INK, fontweight="bold", loc="left")
    ax.legend(frameon=False, labelcolor=INK)
    _save(fig, path)


def plot_comparison(results: dict[str, dict], path, n: int):
    """results: {display_name: metrics_dict}; names containing 'fine' get the highlight colour."""
    metrics = [
        ("schema_valid_pct", "Valid\nschema"),
        ("category_acc_pct", "Category\naccuracy"),
        ("urgency_acc_pct", "Urgency\naccuracy"),
        ("entity_span_f1_grounded_pct", "Entity span F1\n(extractable fields)"),
        ("full_exact_grounded_pct", "Full exact match\n(extractable fields)"),
    ]
    names = list(results)
    colors = []
    for nm in names:
        low = nm.lower()
        colors.append(PALETTE["fine-tuned"] if "fine" in low else PALETTE["tfidf"] if "tf-idf" in low else PALETTE["zero-shot"])

    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=150)
    fig.patch.set_facecolor("white")
    _style(ax)
    width = 0.8 / len(names)
    for i, (nm, col) in enumerate(zip(names, colors)):
        xs = [j + (i - (len(names) - 1) / 2) * width for j in range(len(metrics))]
        vals = [results[nm].get(k) for k, _ in metrics]
        vals_plot = [v if v is not None else 0 for v in vals]
        bars = ax.bar(xs, vals_plot, width * 0.92, label=nm, color=col)
        for b, v in zip(bars, vals):
            label = "n/a" if v is None else f"{v:.0f}"
            ax.text(b.get_x() + b.get_width() / 2, (v or 0) + 1.2, label, ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m[1] for m in metrics], color=INK)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Score (%)", color=INK)
    ax.set_title(f"Bilingual ticket triage: held-out test set (n={n})", color=INK, fontweight="bold", loc="left")
    ax.legend(frameon=False, labelcolor=INK, ncol=len(names), loc="upper center", bbox_to_anchor=(0.5, -0.12))
    _save(fig, path)


def plot_confusion(cm, categories, path, title="Category confusion matrix (fine-tuned)"):
    cols = categories + ["invalid"]
    fig, ax = plt.subplots(figsize=(7.5, 6.2), dpi=150)
    fig.patch.set_facecolor("white")
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", color=INK)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories, color=INK)
    ax.set_xlabel("Predicted", color=INK)
    ax.set_ylabel("Gold", color=INK)
    ax.set_title(title, color=INK, fontweight="bold", loc="left")
    vmax = max(max(r) for r in cm) or 1
    for i, row in enumerate(cm):
        for j, v in enumerate(row):
            if v:
                ax.text(j, i, v, ha="center", va="center", fontsize=9, color="white" if v > vmax * 0.55 else INK)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, path)
