"""Download the bilingual ticket-triage dataset and format it for SFT.

Each row becomes a conversational prompt/completion pair so that TRL computes the
loss on the assistant JSON only (not on the system prompt or the ticket text).
"""

import json
from pathlib import Path

from datasets import load_dataset

from common import ENTITY_KEYS, build_system_prompt

DATASET_ID = "abuzarkhan/bilingual-ticket-triage-dataset"


def normalize_entities(raw) -> dict:
    """Return the entity dict with exactly ENTITY_KEYS, in a fixed order (missing -> None)."""
    raw = raw if isinstance(raw, dict) else {}
    out = {}
    for k in ENTITY_KEYS:
        v = raw.get(k)
        out[k] = v.strip() if isinstance(v, str) and v.strip() else None
    return out


def build_datasets(output_dir: str = "data") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(DATASET_ID)

    categories = sorted(set(ds["train"]["category"]))
    urgencies = sorted(set(ds["train"]["urgency"]))
    sentiments = sorted({e["sentiment"] for e in ds["train"]["entities"]})
    system_prompt = build_system_prompt(categories, urgencies, sentiments)

    meta = {
        "dataset": DATASET_ID,
        "system_prompt": system_prompt,
        "categories": categories,
        "urgencies": urgencies,
        "sentiments": sentiments,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    def _map(row):
        text = str(row["text"]).strip()
        target = {
            "category": str(row["category"]).strip(),
            "urgency": str(row["urgency"]).strip(),
            "entities": normalize_entities(row["entities"]),
        }
        target_json = json.dumps(target, ensure_ascii=False)
        return {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            "completion": [{"role": "assistant", "content": target_json}],
            "input_text": text,
            "ground_truth": target_json,
        }

    for split in ("train", "validation", "test"):
        formatted = ds[split].map(_map, remove_columns=ds[split].column_names)
        formatted.save_to_disk(str(out / split))
        print(f"{split}: {len(formatted)} rows")

    print(f"categories: {categories}\nurgencies: {urgencies}\nsentiments: {sentiments}")


if __name__ == "__main__":
    build_datasets()
