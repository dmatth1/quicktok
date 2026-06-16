"""Importer contract: refusals are catchable ValueErrors (never sys.exit),
data_home honors the env, and (network-gated) a real import round-trips."""
import json
import os
import pytest
import quicktok
from quicktok.importer import ImportRefused, data_home


def _write(tmp_path, obj):
    p = tmp_path / "tokenizer.json"
    p.write_text(json.dumps(obj))
    return str(p)


def test_import_refused_is_value_error():
    # the contract: catchable, not a sys.exit / SystemExit
    assert issubclass(ImportRefused, ValueError)


@pytest.mark.parametrize("obj", [
    {"model": {"type": "Unigram"}},                                   # not BPE
    {"pre_tokenizer": {"type": "Whitespace"}, "model": {"type": "BPE"}},  # not byte-level
    {"model": {"type": "BPE", "byte_fallback": True}},                # byte_fallback unsupported
])
def test_unsupported_tokenizers_are_refused(tmp_path, obj):
    src = _write(tmp_path, obj)
    with pytest.raises(ImportRefused):
        quicktok.import_tokenizer(src, "rejectme", data_dir=str(tmp_path))


def test_import_does_not_sys_exit(tmp_path):
    # regression: importer used to sys.exit; library callers must get an exception
    src = _write(tmp_path, {"model": {"type": "WordPiece"}})
    try:
        quicktok.import_tokenizer(src, "x", data_dir=str(tmp_path))
    except ImportRefused:
        pass
    except SystemExit:
        pytest.fail("import_tokenizer raised SystemExit instead of ImportRefused")


def test_missing_source_raises_catchable(tmp_path):
    with pytest.raises((ImportRefused, ValueError, FileNotFoundError, OSError)):
        quicktok.import_tokenizer(str(tmp_path / "nope.json"), "x", data_dir=str(tmp_path))


def test_data_home_respects_env(monkeypatch, tmp_path):
    monkeypatch.setenv("QUICKTOK_DATA", str(tmp_path / "qt"))
    assert data_home() == str(tmp_path / "qt")
    monkeypatch.delenv("QUICKTOK_DATA", raising=False)
    assert ".cache/quicktok" in data_home() or "quicktok" in data_home()


@pytest.mark.skipif(not os.environ.get("QUICKTOK_NET_TESTS"),
                    reason="network/dep-gated; set QUICKTOK_NET_TESTS=1 (needs tokenizers, network)")
def test_positive_import_roundtrip(tmp_path):
    # end-to-end: download a small ungated byte-level BPE tokenizer, import +
    # verify, then tokenize through the imported encoding.
    pytest.importorskip("tokenizers")
    src = "https://huggingface.co/NousResearch/Meta-Llama-3-8B/resolve/main/tokenizer.json"
    quicktok.import_tokenizer(src, "imptest", data_dir=str(tmp_path), corpus=())
    enc = quicktok.get_encoding("imptest", data_dir=str(tmp_path))
    s = "hello imported tokenizer 123"
    assert enc.decode(enc.encode_ordinary(s)) == s


def test_tekken_synthetic_import_roundtrip(tmp_path):
    """Hermetic positive import: a synthetic tekken.json (real o200k vocab + the
    Tekken v3 pattern) imports, the importer's built-in verification confirms the
    tekken scanner is byte-exact vs the tiktoken reference, and it round-trips.
    Covers the importer positive pipeline AND the pretok_next_tekken scanner."""
    import base64, json
    tiktoken = pytest.importorskip("tiktoken")
    from quicktok._import_core import TEKKEN_PAT
    ranks = tiktoken.get_encoding("o200k_base")._mergeable_ranks
    vocab = [{"rank": r, "token_bytes": base64.b64encode(b).decode()} for b, r in ranks.items()]
    tj = {"config": {"pattern": TEKKEN_PAT, "default_vocab_size": len(ranks),
                     "default_num_special_tokens": 0},
          "vocab": vocab, "special_tokens": []}
    src = tmp_path / "tekken.json"          # filename must contain "tekken"
    src.write_text(json.dumps(tj))
    quicktok.import_tokenizer(str(src), "tekkentest", data_dir=str(tmp_path))
    enc = quicktok.get_encoding("tekkentest", data_dir=str(tmp_path))
    assert enc.n_vocab > 0
    for s in ["Hello WORLD test 123", "café ÀÉÎ lower 42", "punct!!!  \n\n  spaces", "日本語 mix"]:
        assert enc.decode(enc.encode_ordinary(s)) == s, s
