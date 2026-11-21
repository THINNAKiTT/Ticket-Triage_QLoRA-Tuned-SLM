"""Render charts and a markdown results table from results/*.json (no GPU needed)."""

import argparse
import json
from pathlib import Path

from common import load_meta
from plots import plot_comparison, plot_confusion, plot_loss_curve

RESULTS_DIR = Path("results")
ASSETS_DIR = Path("assets")


def fmt(v):
    return "n/a" if v is None else f"{v:.1f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="qwen2.5-0.5b")
    args = p.parse_args()

    meta = load_meta()
    ev = json.loads((RESULTS_DIR / f"eval_{args.tag}.json").read_text())
    r, n = ev["results"], ev["n"]
    base_path = RESULTS_DIR / "baseline_tfidf.json"
    tfidf = json.loads(base_path.read_text()) if base_path.exists() else None

    rows = {
        "Base, zero-shot": r["zero_shot"],
        "Base, 8-shot": r["few_shot"],
    }
    if tfidf:
        rows["TF-IDF + LogReg (no entities)"] = tfidf
    rows["Fine-tuned (QLoRA)"] = r["fine_tuned"]
    plot_comparison(rows, ASSETS_DIR / "eval_comparison.png", n)

    plot_confusion(r["fine_tuned"]["confusion_matrix"], meta["categories"], ASSETS_DIR / "confusion_matrix.png")

    train_path = RESULTS_DIR / f"train_{args.tag}.json"
    if train_path.exists():
        tr = json.loads(train_path.read_text())
        plot_loss_curve(tr["log_history"], ASSETS_DIR / f"loss_curve_{args.tag}.png", title=tr["model"].split("/")[-1])

    all_rows = dict(rows)
    if "fine_tuned_merged_bf16" in r:
        all_rows["Fine-tuned, merged (bf16)"] = r["fine_tuned_merged_bf16"]
    lines = [
        "| System | Schema-valid | Category acc | Urgency acc | Entity F1 (all) | Entity F1 (extractable) | Full exact (all) | Full exact (extractable) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in all_rows.items():
        lines.append(
            f"| {name} | {fmt(m.get('schema_valid_pct'))} | {fmt(m.get('category_acc_pct'))} | "
            f"{fmt(m.get('urgency_acc_pct'))} | {fmt(m.get('entity_span_f1_pct'))} | "
            f"{fmt(m.get('entity_span_f1_grounded_pct'))} | {fmt(m.get('full_exact_pct'))} | "
            f"{fmt(m.get('full_exact_grounded_pct'))} |"
        )
    table = "\n".join(lines)
    (RESULTS_DIR / f"results_table_{args.tag}.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
