"""Shared constants, prompt construction, output parsing and metric helpers.

Deliberately free of torch / transformers imports so it can be unit-tested
and reused by the baseline without a GPU.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

DATA_DIR = Path("data")
META_PATH = DATA_DIR / "meta.json"

ENTITY_KEYS = ("order_id", "product_name", "account_email", "sentiment")
# Entity fields that are free-text spans (null when absent). Sentiment is a closed class.
SPAN_FIELDS = ("order_id", "product_name", "account_email")


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
def build_system_prompt(categories: list[str], urgencies: list[str], sentiments: list[str]) -> str:
    schema = (
        "{"
        f'"category": one of {json.dumps(categories)}, '
        f'"urgency": one of {json.dumps(urgencies)}, '
        '"entities": {'
        '"order_id": string or null, '
        '"product_name": string or null, '
        '"account_email": string or null, '
        f'"sentiment": one of {json.dumps(sentiments)}'
        "}}"
    )
    return (
        "You are a bilingual (English / Roman-Urdu) customer-support ticket triage engine. "
        "Read the ticket and reply with ONLY a single JSON object and nothing else, "
        f"in exactly this shape: {schema}. "
        "Use null for any entity that is not present in the ticket."
    )


def load_meta(path: Path = META_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Output parsing / validation
# --------------------------------------------------------------------------- #
def make_schema(categories: list[str], urgencies: list[str], sentiments: list[str]) -> type[BaseModel]:
    """Build a strict pydantic model for a model response given the label vocabularies."""
    cats, urgs, sents = set(categories), set(urgencies), set(sentiments)

    class Entities(BaseModel):
        model_config = ConfigDict(extra="forbid")
        order_id: str | None
        product_name: str | None
        account_email: str | None
        sentiment: str

        @field_validator("sentiment")
        @classmethod
        def _sentiment(cls, v: str) -> str:
            if v not in sents:
                raise ValueError(f"invalid sentiment: {v}")
            return v

    class Ticket(BaseModel):
        model_config = ConfigDict(extra="forbid")
        category: str
        urgency: str
        entities: Entities

        @field_validator("category")
        @classmethod
        def _category(cls, v: str) -> str:
            if v not in cats:
                raise ValueError(f"invalid category: {v}")
            return v

        @field_validator("urgency")
        @classmethod
        def _urgency(cls, v: str) -> str:
            if v not in urgs:
                raise ValueError(f"invalid urgency: {v}")
            return v

    return Ticket


def parse_response(schema: type[BaseModel], text: str) -> dict[str, Any] | None:
    """Strictly parse a raw model response. Returns a plain dict, or None if invalid.

    "Strict" means the whole response must be exactly one schema-conforming JSON
    object: trailing junk, unknown keys or out-of-vocabulary labels all fail.
    """
    try:
        return schema.model_validate_json(text.strip()).model_dump()
    except ValidationError:
        return None


def is_json(text: str) -> bool:
    try:
        json.loads(text.strip())
        return True
    except (json.JSONDecodeError, ValueError):
        return False


# --------------------------------------------------------------------------- #
# Language heuristic
# --------------------------------------------------------------------------- #
# The dataset has no language column. This is a rough wordlist heuristic, used only
# to slice the evaluation; it is not ground truth.
_ROMAN_URDU = {
    "nahi", "nahin", "hai", "hain", "kya", "mujhe", "mujhay", "mera", "meri", "mere", "maine", "mein",
    "ko", "ka", "ki", "ke", "aap", "aapka", "kar", "karen", "karein", "kardo", "kab", "kyun", "kyu",
    "abhi", "bhi", "se", "par", "pe", "lekin", "magar", "ya", "ye", "yeh", "wo", "woh", "tha", "thi",
    "raha", "rahi", "rahay", "gaya", "gayi", "hua", "hui", "paisay", "paise", "kharab", "jaldi",
    "bhej", "bheja", "diya", "dena", "chahiye", "krna", "krdo", "wapas", "dobara", "aayega", "aya",
}
_WORD = re.compile(r"[a-z']+")


def detect_language(text: str) -> str:
    """Return 'roman_urdu' (incl. code-mixed) or 'english' via a wordlist heuristic."""
    hits = sum(w in _ROMAN_URDU for w in _WORD.findall(text.lower()))
    return "roman_urdu" if hits >= 2 else "english"


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _norm(v: Any) -> Any:
    return v.strip().lower() if isinstance(v, str) else v


def pct_of(flags: list[bool], n: int) -> float:
    return 100.0 * sum(flags) / n


def is_grounded(gold_value: Any, text: str) -> bool:
    """A gold span value is 'grounded' if it is null or appears verbatim in the ticket text.

    The dataset's test split contains many non-null entities (e.g. emails) that never
    appear in the ticket, which no extractor could recover. Grounded metrics ignore
    those fields so extraction quality is not conflated with unrecoverable labels.
    """
    return gold_value is None or str(gold_value).strip().lower() in text.lower()


def score_predictions(
    preds: list[dict[str, Any] | None],
    golds: list[dict[str, Any]],
    json_ok: list[bool],
    categories: list[str],
    texts: list[str] | None = None,
) -> dict[str, Any]:
    """Compute headline + per-class + entity metrics.

    `preds[i]` is None when response i did not satisfy the strict schema; such rows
    count as wrong for every accuracy metric (denominator is always the full set).
    If `texts` is given, additional "grounded" entity metrics are computed (see is_grounded).
    """
    n = len(golds)
    assert len(preds) == n == len(json_ok)

    schema_ok = sum(p is not None for p in preds)
    cat_ok = [p is not None and p["category"] == g["category"] for p, g in zip(preds, golds)]
    urg_ok = [p is not None and p["urgency"] == g["urgency"] for p, g in zip(preds, golds)]
    joint_ok = [c and u for c, u in zip(cat_ok, urg_ok)]

    # Per-field entity correctness (null == null is correct).
    field_ok = {k: [] for k in ENTITY_KEYS}
    for p, g in zip(preds, golds):
        for k in ENTITY_KEYS:
            field_ok[k].append(p is not None and _norm(p["entities"][k]) == _norm(g["entities"][k]))
    entities_exact = [all(field_ok[k][i] for k in ENTITY_KEYS) for i in range(n)]
    full_exact = [j and e for j, e in zip(joint_ok, entities_exact)]

    # Span extraction P/R/F1 over the three nullable span fields (micro-averaged).
    tp = fp = fn = 0
    for p, g in zip(preds, golds):
        for k in SPAN_FIELDS:
            gv = _norm(g["entities"][k])
            pv = _norm(p["entities"][k]) if p is not None else None
            if pv is not None and pv == gv:
                tp += 1
            else:
                if pv is not None:
                    fp += 1
                if gv is not None:
                    fn += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    # Per-class category P/R/F1 (invalid predictions count as a miss for the gold class).
    per_class = {}
    for c in categories:
        c_tp = sum(1 for p, g in zip(preds, golds) if p and p["category"] == c and g["category"] == c)
        c_fp = sum(1 for p, g in zip(preds, golds) if p and p["category"] == c and g["category"] != c)
        c_fn = sum(1 for p, g in zip(preds, golds) if g["category"] == c and not (p and p["category"] == c))
        cp = c_tp / (c_tp + c_fp) if c_tp + c_fp else 0.0
        cr = c_tp / (c_tp + c_fn) if c_tp + c_fn else 0.0
        per_class[c] = {
            "precision": cp,
            "recall": cr,
            "f1": 2 * cp * cr / (cp + cr) if cp + cr else 0.0,
            "support": c_tp + c_fn,
        }
    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(per_class)

    grounded: dict[str, Any] = {}
    if texts is not None:
        assert len(texts) == n
        tp = fp = fn = 0
        ungrounded = nonnull = 0
        ent_exact_g = []
        for p, g, t, rows_ok in zip(preds, golds, texts, zip(*[field_ok[k] for k in ENTITY_KEYS])):
            row_ok = True
            for k, ok in zip(ENTITY_KEYS, rows_ok):
                gv = g["entities"][k]
                if k in SPAN_FIELDS and gv is not None:
                    nonnull += 1
                    if not is_grounded(gv, t):
                        ungrounded += 1
                        continue  # unrecoverable label: skip this field entirely
                if not ok:
                    row_ok = False
                if k in SPAN_FIELDS:
                    gvn = _norm(gv)
                    pv = _norm(p["entities"][k]) if p is not None else None
                    if pv is not None and pv == gvn:
                        tp += 1
                    else:
                        fp += pv is not None
                        fn += gvn is not None
            ent_exact_g.append(row_ok)
        gp = tp / (tp + fp) if tp + fp else 0.0
        gr = tp / (tp + fn) if tp + fn else 0.0
        grounded = {
            "ungrounded_gold_span_pct": 100.0 * ungrounded / nonnull if nonnull else 0.0,
            "entity_span_f1_grounded_pct": 100.0 * (2 * gp * gr / (gp + gr) if gp + gr else 0.0),
            "entity_span_precision_grounded_pct": 100.0 * gp,
            "entity_span_recall_grounded_pct": 100.0 * gr,
            "entities_exact_grounded_pct": pct_of(ent_exact_g, n),
            "full_exact_grounded_pct": pct_of([j and e for j, e in zip(joint_ok, ent_exact_g)], n),
        }

    pct = lambda xs: 100.0 * sum(xs) / n  # noqa: E731
    return {
        **grounded,
        "n": n,
        "valid_json_pct": pct(json_ok),
        "schema_valid_pct": 100.0 * schema_ok / n,
        "category_acc_pct": pct(cat_ok),
        "urgency_acc_pct": pct(urg_ok),
        "joint_acc_pct": pct(joint_ok),
        "category_macro_f1_pct": 100.0 * macro_f1,
        "entity_field_acc_pct": {k: pct(v) for k, v in field_ok.items()},
        "entity_span_f1_pct": 100.0 * f1,
        "entity_span_precision_pct": 100.0 * prec,
        "entity_span_recall_pct": 100.0 * rec,
        "entities_exact_pct": pct(entities_exact),
        "full_exact_pct": pct(full_exact),
        "per_class": per_class,
    }


def confusion_matrix(
    preds: list[dict[str, Any] | None], golds: list[dict[str, Any]], categories: list[str]
) -> list[list[int]]:
    """Rows = gold, columns = predicted categories + a final 'invalid' column."""
    idx = {c: i for i, c in enumerate(categories)}
    m = [[0] * (len(categories) + 1) for _ in categories]
    for p, g in zip(preds, golds):
        m[idx[g["category"]]][idx[p["category"]] if p else len(categories)] += 1
    return m


def slice_by_language(texts: list[str]) -> Counter:
    return Counter(detect_language(t) for t in texts)
