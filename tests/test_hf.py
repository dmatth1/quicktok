"""Tests for the HuggingFace AutoTokenizer drop-in (quicktok.patch_transformers).

The wrapper fast-paths the encode hot path through quicktok when the model's
grammar is supported, and delegates everything else to the real HF tokenizer.

Unit tests use a small *real* HF-shaped collaborator (`FakeHF`) whose ordinary
encoding is raw UTF-8 bytes — deliberately different from quicktok's BPE ids —
so the fast-path-vs-delegate decision is provable from the output. A
transformers-gated integration test exercises the real AutoTokenizer patch.
"""
import pytest
import quicktok
from conftest import SAMPLES
from quicktok.hf import (
    QuicktokBackedTokenizer,
    wrap_pretrained,
    patch_transformers,
    unpatch_transformers,
)

# Open (Apache-2.0), small, qwen-grammar model — quicktok backs it; used for the
# real end-to-end exactness check against transformers.AutoTokenizer.
INTEGRATION_MODEL = "Qwen/Qwen2.5-0.5B"


@pytest.fixture(scope="module")
def qt():
    return quicktok.get_encoding("cl100k_base")


class FakeHF:
    """Minimal real stand-in for a HF fast tokenizer. Ordinary encoding is raw
    UTF-8 byte values (distinct from quicktok BPE ids, so the wrapper's path is
    visible in the output); `add_special_tokens=True` applies a wrap that is
    either a static [bos]..[eos] or a content-dependent prefix (to exercise the
    fall-back path)."""

    def __init__(self, bos=1, eos=2, specials="static"):
        self.bos, self.eos, self.specials = bos, eos, specials
        self.name_or_path = "fake/model"

    def encode(self, text, add_special_tokens=True, **kw):
        ids = list(text.encode("utf-8"))
        if add_special_tokens:
            if self.specials == "static":
                return [self.bos] + ids + [self.eos]
            if self.specials == "content":            # not a static prefix/suffix
                return [len(text)] + ids
        return ids

    def convert_ids_to_tokens(self, ids):             # an arbitrary non-encode method
        return ["<tok>"] * len(ids)


def test_encode_without_specials_uses_quicktok(qt):
    w = wrap_pretrained(FakeHF(), qt)
    text = "hello world, BPE!"
    assert w.encode(text, add_special_tokens=False) == qt.encode_ordinary(text)
    assert w.encode(text, add_special_tokens=False) != list(text.encode())  # not the HF byte path


def test_encode_default_specials_wrap_quicktok_ids(qt):
    w = wrap_pretrained(FakeHF(bos=1, eos=2), qt)
    text = "café résumé"
    assert w.encode(text) == [1] + qt.encode_ordinary(text) + [2]


def test_unsupported_kwargs_delegate(qt):
    w = wrap_pretrained(FakeHF(), qt)
    text = "delegate me"
    # truncation is not something the fast path reproduces -> must delegate to HF
    assert w.encode(text, add_special_tokens=False, truncation=True) == list(text.encode())


def test_getattr_delegates(qt):
    w = wrap_pretrained(FakeHF(), qt)
    assert w.convert_ids_to_tokens([0, 0, 0]) == ["<tok>", "<tok>", "<tok>"]
    assert w.name_or_path == "fake/model"


def test_content_dependent_specials_fall_back(qt):
    w = wrap_pretrained(FakeHF(specials="content"), qt)
    text = "abc"
    # specials are not a static prefix/suffix -> add_special_tokens=True must delegate
    assert w.encode(text) == [len(text)] + list(text.encode())


def test_call_returns_input_ids(qt):
    w = wrap_pretrained(FakeHF(), qt)
    text = "tokenize this"
    out = w(text, add_special_tokens=False)
    assert out["input_ids"] == qt.encode_ordinary(text)
    assert out["attention_mask"] == [1] * len(out["input_ids"])


def test_unsupported_model_passes_through(qt):
    # qt=None means quicktok cannot do this model -> wrapper must be transparent
    hf = FakeHF()
    assert wrap_pretrained(hf, None) is hf


def test_patch_without_transformers_raises():
    try:
        import transformers  # noqa: F401
        pytest.skip("transformers installed; covered by the integration test")
    except ImportError:
        pass
    with pytest.raises(ImportError, match="transformers"):
        patch_transformers()


def test_public_api_exports():
    # README/docs use quicktok.patch_transformers(); they must be exported
    assert quicktok.patch_transformers is patch_transformers
    assert quicktok.unpatch_transformers is unpatch_transformers


def _is_patched(transformers):
    return getattr(transformers.AutoTokenizer.from_pretrained, "_quicktok_patched", False)


def test_patch_and_unpatch_swap_from_pretrained():
    transformers = pytest.importorskip("transformers")
    assert not _is_patched(transformers)
    patch_transformers()
    try:
        assert _is_patched(transformers)
        patch_transformers()                  # idempotent
        assert _is_patched(transformers)
    finally:
        unpatch_transformers()
    assert not _is_patched(transformers)


def test_integration_autotokenizer_byte_exact():
    """End-to-end through the real transformers.AutoTokenizer: a patched
    from_pretrained must produce byte-identical ids to the unmodified HF
    tokenizer. Network-gated (downloads the model's tokenizer); skips offline."""
    transformers = pytest.importorskip("transformers")
    try:
        ref = transformers.AutoTokenizer.from_pretrained(INTEGRATION_MODEL)  # unpatched reference
    except Exception as e:  # offline / hub error -> not a quicktok failure
        pytest.skip(f"could not fetch {INTEGRATION_MODEL}: {type(e).__name__}: {e}")

    patch_transformers()
    try:
        fast = transformers.AutoTokenizer.from_pretrained(INTEGRATION_MODEL)
        assert isinstance(fast, QuicktokBackedTokenizer)            # quicktok actually backs it
        for s in SAMPLES:
            assert fast.encode(s, add_special_tokens=False) == ref.encode(s, add_special_tokens=False), s
            assert fast.encode(s) == ref.encode(s), s              # default (specials) path too
        assert list(fast("hi there")["input_ids"]) == list(ref("hi there")["input_ids"])
    finally:
        unpatch_transformers()
