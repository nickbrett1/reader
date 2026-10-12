"""Embedding client: data-URI wrapping, no-mixing rule, batch splitting."""

from __future__ import annotations

import base64

import pytest

from reader.embed import (
    EmbedClient,
    ImageInput,
    MixedModalityError,
    TextInput,
    data_uri,
    image_cost_usd,
    split_batches,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"payload"


def test_data_uri_wraps_as_image():
    uri = data_uri("image/png", PNG)
    assert uri.startswith("data:image/png;base64,")
    assert uri.endswith(base64.b64encode(PNG).decode())
    # Not bare base64.
    assert not uri.startswith(base64.b64encode(PNG).decode()[:8])


def test_data_uri_rejects_non_image():
    with pytest.raises(ValueError):
        data_uri("text/plain", b"hi")


def test_mixed_modality_rejected_client_side():
    client = EmbedClient("http://gw", "k", "m")
    with pytest.raises(MixedModalityError):
        client.embed([TextInput("hi"), ImageInput("image/png", PNG)])


def test_split_batches_by_count():
    items = [TextInput(str(i)) for i in range(5)]
    batches = split_batches(items, max_batch=2, max_bytes=10**9)
    assert [len(b) for b in batches] == [2, 2, 1]


def test_split_batches_by_bytes():
    items = [TextInput("x" * 10) for _ in range(4)]
    batches = split_batches(items, max_batch=100, max_bytes=25)
    assert [len(b) for b in batches] == [2, 2]


def test_embed_texts_orders_by_index_and_wraps_images():
    calls = []

    def fake_http(payload):
        calls.append(payload)
        n = len(payload["input"])
        # Deliberately return rows out of order.
        return {"data": [{"index": i, "embedding": [float(i)] * 4} for i in reversed(range(n))]}

    client = EmbedClient("http://gw", "k", "cohere/embed-v5.0", http=fake_http)
    res = client.embed_texts(["a", "b", "c"])
    assert [v[0] for v in res.vectors] == [0.0, 1.0, 2.0]
    assert res.dimensions == 4
    assert res.provenance()["model_id"] == "cohere/embed-v5.0"

    calls.clear()
    img_res = client.embed_images([("image/png", PNG)])
    assert len(img_res.vectors) == 1
    assert calls[0]["input"][0].startswith("data:image/png;base64,")


def test_empty_input_sends_nothing():
    calls = []
    client = EmbedClient("http://gw", "k", "m", http=lambda p: calls.append(p) or {})
    assert client.embed([]).vectors == []
    assert calls == []


def test_image_cost_at_list_rate():
    # 1M image tokens at $0.40/Mtok.
    assert image_cost_usd(1_000_000) == pytest.approx(0.40)
    assert image_cost_usd(0) == 0.0
