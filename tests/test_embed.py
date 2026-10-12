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


def _png_bytes(w: int, h: int) -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (w, h), color="white").save(buf, format="PNG")
    return buf.getvalue()


def _jpeg_bytes(w: int, h: int) -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (w, h), color="white").save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def test_oversized_image_is_resized_to_cap_and_stays_valid():
    from PIL import Image

    from reader.embed import resize_image
    from reader.schema import IMAGE_MAX_PIXELS

    # 3000 x 1000 = 3.0 Mpx, over the 2.4 Mpx cap.
    data = _png_bytes(3000, 1000)
    out = resize_image("image/png", data)
    assert out != data
    import io

    with Image.open(io.BytesIO(out)) as im:
        assert im.format == "PNG"
        assert im.width * im.height <= IMAGE_MAX_PIXELS


def test_oversized_jpeg_reencoded_as_jpeg():
    from PIL import Image

    from reader.embed import resize_image
    from reader.schema import IMAGE_MAX_PIXELS

    data = _jpeg_bytes(3000, 1000)
    out = resize_image("image/jpeg", data)
    assert out != data
    import io

    with Image.open(io.BytesIO(out)) as im:
        assert im.format == "JPEG"
        assert im.width * im.height <= IMAGE_MAX_PIXELS


def test_under_cap_image_passed_through_byte_for_byte():
    from reader.embed import resize_image
    from reader.schema import IMAGE_MAX_PIXELS

    data = _png_bytes(500, 500)  # 0.25 Mpx, well under the cap
    assert 500 * 500 <= IMAGE_MAX_PIXELS
    assert resize_image("image/png", data) is data
    # ...and the wrap path embeds the original bytes, unmodified.
    client = EmbedClient("http://gw", "k", "m", http=lambda p: {"data": []})
    uri = client._to_input(ImageInput("image/png", data))
    assert uri.endswith(base64.b64encode(data).decode())


def test_resize_preserves_aspect_ratio_within_one_pixel():
    from PIL import Image

    from reader.embed import resize_image

    w, h = 4000, 1000  # 4.0 Mpx, 4:1
    data = _png_bytes(w, h)
    out = resize_image("image/png", data)
    import io

    with Image.open(io.BytesIO(out)) as im:
        nw, nh = im.width, im.height
    assert nw * nh <= 2_400_000
    # Same aspect ratio to within ~1px of the longer edge.
    assert abs(nw - (w / h) * nh) <= 1



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


def test_split_batches_by_pixels():
    import io

    from PIL import Image

    def make_png(w: int, h: int) -> bytes:
        img = Image.new("RGB", (w, h), color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    # 3 images: 1000x1000 (1MP each)
    p1 = make_png(1000, 1000)
    items = [ImageInput("image/png", p1) for _ in range(3)]
    # Cap at 2.5 MP: should yield [2, 1] batches
    batches = split_batches(items, max_batch=100, max_bytes=10**9, max_pixels=2_500_000)
    assert [len(b) for b in batches] == [2, 1]


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
    # Rule 3: image accounting is flagged unreliable.
    assert img_res.usage["image_tokens_reliable"] is False


def test_empty_input_sends_nothing():
    calls = []
    client = EmbedClient("http://gw", "k", "m", http=lambda p: calls.append(p) or {})
    assert client.embed([]).vectors == []
    assert calls == []


def test_image_cost_at_list_rate():
    # 1M image tokens at $0.40/Mtok.
    assert image_cost_usd(1_000_000) == pytest.approx(0.40)
    assert image_cost_usd(0) == 0.0
