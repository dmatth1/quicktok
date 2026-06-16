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
