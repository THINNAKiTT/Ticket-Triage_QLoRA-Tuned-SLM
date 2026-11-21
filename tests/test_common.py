import json

import pytest

from common import (
    build_system_prompt,
    confusion_matrix,
    detect_language,
    make_schema,
    parse_response,
    score_predictions,
)

CATS = ["billing", "other", "technical_issue"]
URGS = ["high", "low", "medium"]
SENTS = ["negative", "neutral", "positive"]
Schema = make_schema(CATS, URGS, SENTS)


def ticket(category="billing", urgency="high", **ent):
    e = {"order_id": None, "product_name": None, "account_email": None, "sentiment": "neutral"}
    e.update(ent)
    return {"category": category, "urgency": urgency, "entities": e}


def test_system_prompt_lists_every_label_and_entity_key():
    p = build_system_prompt(CATS, URGS, SENTS)
    for token in CATS + URGS + SENTS + ["order_id", "product_name", "account_email", "sentiment"]:
        assert token in p


def test_parse_accepts_a_valid_response():
    assert parse_response(Schema, json.dumps(ticket())) == ticket()


@pytest.mark.parametrize(
    "bad",
    [
        "not json",
        json.dumps(ticket()) + ")",  # trailing junk, a failure mode seen in the first adapter
        json.dumps({**ticket(), "category": "payment_incomplete"}),  # hallucinated label
        json.dumps({**ticket(), "urgency": "urgent"}),
        json.dumps({"category": "billing", "urgency": "high", "entities": {}}),  # missing entity keys
        json.dumps({**ticket(), "entities": {**ticket()["entities"], "price": "$5"}}),  # invented key
        json.dumps({**ticket(), "extra": 1}),
    ],
)
def test_parse_rejects_invalid_responses(bad):
    assert parse_response(Schema, bad) is None


def test_score_counts_invalid_as_wrong_over_full_denominator():
    golds = [ticket(), ticket("other", "low")]
    preds = [ticket(), None]
    m = score_predictions(preds, golds, [True, False], CATS)
    assert m["schema_valid_pct"] == 50.0
    assert m["category_acc_pct"] == 50.0
    assert m["joint_acc_pct"] == 50.0
    assert m["full_exact_pct"] == 50.0


def test_entity_scoring_null_matches_null_and_penalises_hallucination():
    golds = [ticket(order_id="ORD-1"), ticket()]
    # row 0: correct order id; row 1: hallucinated an email where gold is null
    preds = [ticket(order_id="ORD-1"), ticket(account_email="info@example.com")]
    m = score_predictions(preds, golds, [True, True], CATS)
    assert m["entity_field_acc_pct"]["order_id"] == 100.0
    assert m["entity_field_acc_pct"]["account_email"] == 50.0
    # tp=1 (order id), fp=1 (invented email), fn=0
    assert m["entity_span_precision_pct"] == 50.0
    assert m["entity_span_recall_pct"] == 100.0
    assert m["entities_exact_pct"] == 50.0


def test_entity_match_is_case_and_whitespace_insensitive():
    golds = [ticket(product_name="Smart Thermostat")]
    preds = [ticket(product_name="  smart thermostat ")]
    m = score_predictions(preds, golds, [True], CATS)
    assert m["entity_field_acc_pct"]["product_name"] == 100.0


def test_per_class_and_confusion_matrix():
    golds = [ticket("billing"), ticket("billing"), ticket("other")]
    preds = [ticket("billing"), ticket("other"), None]
    m = score_predictions(preds, golds, [True, True, False], CATS)
    assert m["per_class"]["billing"]["recall"] == 0.5
    assert m["per_class"]["billing"]["precision"] == 1.0
    cm = confusion_matrix(preds, golds, CATS)
    assert cm[0] == [1, 1, 0, 0]  # billing -> billing, other
    assert cm[1] == [0, 0, 0, 1]  # other -> invalid


def test_language_heuristic():
    assert detect_language("The app keeps crashing when I scroll.") == "english"
    assert detect_language("mujhse extra paisay kat gaye, ye galat charge hai") == "roman_urdu"


def test_grounded_metrics_ignore_unrecoverable_gold_fields():
    from common import is_grounded

    assert is_grounded(None, "anything")
    assert is_grounded("ORD-1", "my order ord-1 is late")
    assert not is_grounded("a@b.com", "no email in this ticket")

    # Gold email is not in the text -> model correctly says null, but raw metrics call it wrong.
    golds = [ticket(account_email="a@b.com", order_id="ORD-1")]
    preds = [ticket(order_id="ORD-1")]
    texts = ["where is order ORD-1"]
    m = score_predictions(preds, golds, [True], CATS, texts=texts)
    assert m["entity_span_f1_pct"] < 100.0
    assert m["entity_span_f1_grounded_pct"] == 100.0
    assert m["entities_exact_pct"] == 0.0
    assert m["entities_exact_grounded_pct"] == 100.0
    assert m["ungrounded_gold_span_pct"] == 50.0
