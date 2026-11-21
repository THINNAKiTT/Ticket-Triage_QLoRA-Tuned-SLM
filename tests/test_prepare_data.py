from prepare_data import normalize_entities


def test_normalize_entities_fixed_keys_and_order():
    out = normalize_entities({"sentiment": "negative", "order_id": " ORD-1 ", "extra": "x"})
    assert list(out) == ["order_id", "product_name", "account_email", "sentiment"]
    assert out["order_id"] == "ORD-1"
    assert out["product_name"] is None


def test_normalize_entities_handles_garbage():
    assert normalize_entities(None)["sentiment"] is None
    assert normalize_entities("oops")["order_id"] is None
    assert normalize_entities({"order_id": "  "})["order_id"] is None
