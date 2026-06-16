"""encode_batch (flat tokens+offsets arrays) and count_batch."""
import pytest
import quicktok
from conftest import SAMPLES

np = pytest.importorskip("numpy")


@pytest.fixture
def enc():
    return quicktok.get_encoding("cl100k_base")


def test_encode_batch_flat_matches_per_text(enc):
    texts = [s for s in SAMPLES if s]
    tokens, offsets = enc.encode_batch(texts)
    assert offsets.dtype == np.int64 and tokens.dtype == np.uint32
    assert len(offsets) == len(texts) + 1
    assert offsets[0] == 0 and offsets[-1] == len(tokens)
    for i, s in enumerate(texts):
        piece = list(tokens[offsets[i]:offsets[i + 1]])
        assert piece == enc.encode_ordinary(s), s


def test_encode_batch_threads_invariant(enc):
    texts = [f"doc {i}: the quick brown fox 日本語 {i}" for i in range(300)]
    a = enc.encode_batch(texts, 1)
    b = enc.encode_batch(texts, 8)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_encode_batch_with_special(enc):
    texts = ["a <|endoftext|> b", "plain text"]
    tokens, offsets = enc.encode_batch(texts, 0, True)  # with_special=True
    first = list(tokens[offsets[0]:offsets[1]])
    assert first == enc.encode_with_special(texts[0])


def test_encode_batch_empty():
    enc = quicktok.get_encoding("cl100k_base")
    tokens, offsets = enc.encode_batch([])
    assert len(tokens) == 0 and list(offsets) == [0]


def test_count_and_count_batch(enc):
    texts = [s for s in SAMPLES if s]
    counts = quicktok.count_batch(enc, texts)
    assert list(counts) == [enc.count(s) for s in texts]
    assert all(enc.count(s) == len(enc.encode_ordinary(s)) for s in texts)
