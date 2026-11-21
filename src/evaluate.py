"""Evaluate base (zero-/few-shot) vs fine-tuned model on the held-out test split.

Writes results/eval_<tag>.json (metrics) and results/predictions_<tag>.jsonl (raw outputs).
Charts are rendered separately by report.py so they can be regenerated without a GPU.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from common import (
    confusion_matrix,
    detect_language,
    is_json,
    load_meta,
    make_schema,
    parse_response,
    score_predictions,
)

RESULTS_DIR = Path("results")
N_SHOT_PER_CLASS = 1  # few-shot baseline: one train example per category


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--adapter", default="outputs/qwen2.5-0.5b-triage-lora")
    p.add_argument("--merged", default=None, help="optional merged-model dir to evaluate in bf16")
    p.add_argument("--tag", default="qwen2.5-0.5b")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--limit", type=int, default=None, help="evaluate only the first N test rows (debug)")
    p.add_argument("--rescore", action="store_true", help="re-score saved predictions; no model/GPU needed")
    return p.parse_args()


def build_few_shot(train_ds, categories):
    """One deterministic example per category, as extra user/assistant turns."""
    shots, seen = [], defaultdict(int)
    for row in train_ds:
        gold = json.loads(row["ground_truth"])
        c = gold["category"]
        if seen[c] < N_SHOT_PER_CLASS:
            seen[c] += 1
            shots += [
                {"role": "user", "content": row["input_text"]},
                {"role": "assistant", "content": row["ground_truth"]},
            ]
        if len(seen) == len(categories) and all(v >= N_SHOT_PER_CLASS for v in seen.values()):
            break
    return shots


@torch.no_grad()
def generate_all(model, tokenizer, texts, system_prompt, shots, batch_size, max_new_tokens):
    model.eval()
    tokenizer.padding_side = "left"
    outs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt}, *shots, {"role": "user", "content": t}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for t in batch
        ]
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id
        )
        outs += tokenizer.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        print(f"  {min(i + batch_size, len(texts))}/{len(texts)}", end="\r", flush=True)
    print()
    return outs


def evaluate(name, raw_outputs, test_ds, schema, categories):
    golds = [json.loads(g) for g in test_ds["ground_truth"]]
    texts = test_ds["input_text"]
    json_ok = [is_json(r) for r in raw_outputs]
    preds = [parse_response(schema, r) for r in raw_outputs]

    metrics = score_predictions(preds, golds, json_ok, categories, texts=texts)
    metrics["confusion_matrix"] = confusion_matrix(preds, golds, categories)

    # Language slice (heuristic, see common.detect_language).
    langs = [detect_language(t) for t in texts]
    by_lang = {}
    for lang in ("english", "roman_urdu"):
        idx = [i for i, l in enumerate(langs) if l == lang]
        if idx:
            m = score_predictions(
                [preds[i] for i in idx], [golds[i] for i in idx], [json_ok[i] for i in idx], categories,
                texts=[texts[i] for i in idx],
            )
            by_lang[lang] = {
                k: m[k]
                for k in ("n", "category_acc_pct", "urgency_acc_pct", "entity_span_f1_grounded_pct", "full_exact_grounded_pct")
            }
    metrics["by_language"] = by_lang

    print(
        f"[{name}] valid JSON {metrics['valid_json_pct']:.1f}% | schema {metrics['schema_valid_pct']:.1f}% | "
        f"cat {metrics['category_acc_pct']:.1f}% | urg {metrics['urgency_acc_pct']:.1f}% | "
        f"entity F1 {metrics['entity_span_f1_pct']:.1f}% (grounded {metrics['entity_span_f1_grounded_pct']:.1f}%) | "
        f"full exact {metrics['full_exact_pct']:.1f}% (grounded {metrics['full_exact_grounded_pct']:.1f}%)"
    )
    records = [
        {"system": name, "input": t, "gold": g, "raw_output": r, "parsed": p, "language": l}
        for t, g, r, p, l in zip(texts, golds, raw_outputs, preds, langs)
    ]
    return metrics, records


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)
    meta = load_meta()
    categories = meta["categories"]
    schema = make_schema(categories, meta["urgencies"], meta["sentiments"])
    system_prompt = meta["system_prompt"]

    test_ds = load_from_disk("data/test")
    if args.limit:
        test_ds = test_ds.select(range(args.limit))
    train_ds = load_from_disk("data/train")
    shots = build_few_shot(train_ds, categories)
    texts = test_ds["input_text"]

    if args.rescore:
        saved = [json.loads(l) for l in open(RESULTS_DIR / f"predictions_{args.tag}.jsonl", encoding="utf-8")]
        names = {"zero-shot": "zero_shot", "few-shot": "few_shot", "fine-tuned": "fine_tuned", "merged-bf16": "fine_tuned_merged_bf16"}
        results, all_records = {}, []
        for sysname, key in names.items():
            outs = [r["raw_output"] for r in saved if r["system"] == sysname]
            if outs:
                results[key], recs = evaluate(sysname, outs, test_ds, schema, categories)
                all_records += recs
        payload = {"model": args.model, "adapter": args.adapter, "n": len(texts), "results": results}
        (RESULTS_DIR / f"eval_{args.tag}.json").write_text(json.dumps(payload, indent=2))
        with open(RESULTS_DIR / f"predictions_{args.tag}.jsonl", "w", encoding="utf-8") as f:
            for r in all_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Re-scored -> results/eval_{args.tag}.json")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb, device_map="auto", dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, args.adapter)

    results, all_records = {}, []
    gen_kwargs = dict(batch_size=args.batch_size, max_new_tokens=args.max_new_tokens)

    # Same 4-bit base, adapter switched off -> a true "before fine-tuning" comparison.
    with model.disable_adapter():
        print("Base model, zero-shot ...")
        out = generate_all(model, tokenizer, texts, system_prompt, [], **gen_kwargs)
        results["zero_shot"], recs = evaluate("zero-shot", out, test_ds, schema, categories)
        all_records += recs

        print(f"Base model, {len(shots) // 2}-shot ...")
        out = generate_all(model, tokenizer, texts, system_prompt, shots, **gen_kwargs)
        results["few_shot"], recs = evaluate("few-shot", out, test_ds, schema, categories)
        all_records += recs

    print("Fine-tuned (QLoRA adapter) ...")
    out = generate_all(model, tokenizer, texts, system_prompt, [], **gen_kwargs)
    results["fine_tuned"], recs = evaluate("fine-tuned", out, test_ds, schema, categories)
    all_records += recs

    if args.merged:
        del model, base
        torch.cuda.empty_cache()
        print("Merged model (bf16) ...")
        merged = AutoModelForCausalLM.from_pretrained(args.merged, dtype=torch.bfloat16, device_map="auto")
        out = generate_all(merged, tokenizer, texts, system_prompt, [], **gen_kwargs)
        results["fine_tuned_merged_bf16"], recs = evaluate("merged-bf16", out, test_ds, schema, categories)
        all_records += recs

    payload = {"model": args.model, "adapter": args.adapter, "n": len(texts), "results": results}
    (RESULTS_DIR / f"eval_{args.tag}.json").write_text(json.dumps(payload, indent=2))
    with open(RESULTS_DIR / f"predictions_{args.tag}.jsonl", "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote results/eval_{args.tag}.json and results/predictions_{args.tag}.jsonl")


if __name__ == "__main__":
    main()
