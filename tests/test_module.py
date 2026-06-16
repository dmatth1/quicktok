"""Module-level surface: get_encoding, encoding_for_model, bundled loads."""
import pytest
import quicktok
from conftest import BUNDLED, SAMPLES


@pytest.mark.parametrize("name", BUNDLED)
def test_bundled_encoding_loads_and_roundtrips(name):
    enc = quicktok.get_encoding(name)
    assert enc.n_vocab > 0
    for s in SAMPLES:
        assert enc.decode(enc.encode_ordinary(s)) == s, (name, s)


def test_get_encoding_caches():
    assert quicktok.get_encoding("cl100k_base") is quicktok.get_encoding("cl100k_base")


def test_get_encoding_unknown_raises():
    with pytest.raises((RuntimeError, ValueError, KeyError)):
        quicktok.get_encoding("no_such_encoding_xyz")


@pytest.mark.parametrize("model,expected", [
    ("gpt-4o", "o200k_base"),
    ("gpt-4o-mini", "o200k_base"),
    ("gpt-4", "cl100k_base"),
    ("gpt-3.5-turbo", "cl100k_base"),
    ("gpt-oss", "o200k_harmony"),
    ("qwen2.5", "qwen3"),
    ("meta-llama/Llama-3.1-8B", "llama3"),   # HF-style id, org-prefixed, cased
])
def test_encoding_for_model_resolves(model, expected):
    assert quicktok.encoding_for_model(model).name == expected


def test_encoding_for_model_unknown_raises():
    with pytest.raises(KeyError):
        quicktok.encoding_for_model("totally-unknown-model-9000")


def test_version_present():
    assert quicktok.__version__.count(".") == 2
