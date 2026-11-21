"""Classical baseline: TF-IDF (word + char n-grams) + logistic regression.

Predicts category and urgency only (no entity extraction), so it shows what a
trivial non-LLM classifier achieves on the same train/val/test split.
Hyperparameters are picked on the validation split; the test split is scored once.
"""

import json
from pathlib import Path

from datasets import load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import FeatureUnion, make_pipeline

from common import detect_language, load_meta

RESULTS_DIR = Path("results")


def make_model(C: float):
    feats = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), lowercase=True, sublinear_tf=True)),
        ]
    )
    return make_pipeline(feats, LogisticRegression(C=C, max_iter=2000))


def load_split(name):
    ds = load_from_disk(f"data/{name}")
    gold = [json.loads(g) for g in ds["ground_truth"]]
    return ds["input_text"], [g["category"] for g in gold], [g["urgency"] for g in gold]


def fit_best(X_tr, y_tr, X_va, y_va):
    best = None
    for C in (1, 3, 10, 30, 100):
        m = make_model(C).fit(X_tr, y_tr)
        acc = accuracy_score(y_va, m.predict(X_va))
        if best is None or acc > best[0]:
            best = (acc, C, m)
    return best


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    meta = load_meta()
    X_tr, cat_tr, urg_tr = load_split("train")
    X_va, cat_va, urg_va = load_split("validation")
    X_te, cat_te, urg_te = load_split("test")

    _, c_cat, m_cat = fit_best(X_tr, cat_tr, X_va, cat_va)
    _, c_urg, m_urg = fit_best(X_tr, urg_tr, X_va, urg_va)
    p_cat, p_urg = m_cat.predict(X_te), m_urg.predict(X_te)

    cat_ok = [p == g for p, g in zip(p_cat, cat_te)]
    urg_ok = [p == g for p, g in zip(p_urg, urg_te)]
    n = len(X_te)
    langs = [detect_language(t) for t in X_te]
    by_lang = {}
    for lang in ("english", "roman_urdu"):
        idx = [i for i, l in enumerate(langs) if l == lang]
        if idx:
            by_lang[lang] = {
                "n": len(idx),
                "category_acc_pct": 100 * sum(cat_ok[i] for i in idx) / len(idx),
                "urgency_acc_pct": 100 * sum(urg_ok[i] for i in idx) / len(idx),
            }

    out = {
        "n": n,
        "chosen_C": {"category": c_cat, "urgency": c_urg},
        "category_acc_pct": 100 * sum(cat_ok) / n,
        "urgency_acc_pct": 100 * sum(urg_ok) / n,
        "joint_acc_pct": 100 * sum(c and u for c, u in zip(cat_ok, urg_ok)) / n,
        "category_macro_f1_pct": 100 * f1_score(cat_te, p_cat, average="macro", labels=meta["categories"]),
        "by_language": by_lang,
    }
    (RESULTS_DIR / "baseline_tfidf.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "by_language"}, indent=2))


if __name__ == "__main__":
    main()
