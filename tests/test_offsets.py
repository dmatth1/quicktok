"""Per-token offsets: enc.encode_with_offsets(text) -> (ids, spans).

Byte spans are exact by construction — ordinary encode round-trips losslessly,
so each token covers a contiguous byte range and the ranges tile the input.
`unit="char"` reports code-point spans (HF offset_mapping shape).
"""
import pytest
import quicktok
from conftest import SAMPLES


@pytest.fixture(scope="module")
def enc():
    return quicktok.get_encoding("cl100k_base")


@pytest.mark.parametrize("s", [s for s in SAMPLES if s])
def test_byte_spans_tile_and_are_exact(enc, s):
    ids, spans = enc.encode_with_offsets(s)
    b = s.encode("utf-8")
    assert ids == enc.encode_ordinary(s)            # same ids as ordinary encode
    assert len(spans) == len(ids)
    assert spans[0][0] == 0
    assert spans[-1][1] == len(b)                   # tiles the whole input
    for i, (lo, hi) in enumerate(spans):
        assert lo <= hi
        if i:
            assert lo == spans[i - 1][1]            # contiguous, no gaps/overlap
        assert enc.decode_bytes([ids[i]]) == b[lo:hi]   # the span IS the token's bytes


def test_byte_spans_ascii_reconstruct(enc):
    s = "hello world"
    ids, spans = enc.encode_with_offsets(s)
    assert "".join(s[lo:hi] for lo, hi in spans) == s


def test_empty(enc):
    assert enc.encode_with_offsets("") == ([], [])


def test_char_spans_ascii_equal_byte_spans(enc):
    s = "Hello, world! 123 def f(x): return x"
    _, byte_spans = enc.encode_with_offsets(s, unit="byte")
    _, char_spans = enc.encode_with_offsets(s, unit="char")
    assert byte_spans == char_spans                 # ASCII: 1 byte == 1 code point


def test_char_spans_multibyte_index_codepoints(enc):
    s = "café 日本語"
    ids, char_spans = enc.encode_with_offsets(s, unit="char")
    assert char_spans[0][0] == 0
    assert char_spans[-1][1] == len(s)              # code-point length, not byte length
    prev = 0
    for lo, hi in char_spans:                       # monotonic, contiguous in code points
        assert lo == prev and lo <= hi
        prev = hi
    assert "".join(s[lo:hi] for lo, hi in char_spans) == s


def test_invalid_unit_raises(enc):
    with pytest.raises(ValueError):
        enc.encode_with_offsets("x", unit="bogus")


def test_char_offsets_match_huggingface():
    """unit='char' must reproduce HF's return_offsets_mapping byte-for-byte.
    Network-gated (downloads the Qwen tokenizer); skips offline."""
    transformers = pytest.importorskip("transformers")
    qt = quicktok.get_encoding("qwen3")
    try:
        hf = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    except Exception as e:
        pytest.skip(f"could not fetch Qwen tokenizer: {type(e).__name__}: {e}")
    for s in ["hello world foo", "def f(x): return x*2  # ok",
              "café 日本語 test", "ALLCAPS 123 a.b.c", "  spaces\tand\nlines  "]:
        ids, spans = qt.encode_with_offsets(s, unit="char")
        hf_map = [tuple(x) for x in hf(s, return_offsets_mapping=True)["offset_mapping"]]
        assert ids == hf.encode(s, add_special_tokens=False), s
        assert spans == hf_map, s          # char offsets == HF offset_mapping
