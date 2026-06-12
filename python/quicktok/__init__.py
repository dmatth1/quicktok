"""quicktok — fast exact BPE tokenizer.

Bundled encodings: cl100k_base, o200k_base, o200k_harmony, llama3, qwen3.
Drop-in-shaped for tiktoken:

    import quicktok
    enc = quicktok.get_encoding("cl100k_base")
    ids = enc.encode("hello world")          # == tiktoken.encode_ordinary
    text = enc.decode(ids)
"""
import os as _os
from ._quicktok import Tokenizer, _set_datadir, __version__

_set_datadir(_os.path.join(_os.path.dirname(__file__), "data"))

_CACHE = {}

# model name -> encoding (the common ones). encoding_for_model lowercases and
# strips an org prefix, so HF-style ids like "meta-llama/Llama-3.1-8B" resolve.
MODEL_TO_ENCODING = {
    "gpt-4o": "o200k_base", "gpt-4o-mini": "o200k_base", "o1": "o200k_base",
    "o3": "o200k_base", "gpt-4": "cl100k_base", "gpt-4-turbo": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base", "gpt-oss": "o200k_harmony",
    "qwen3": "qwen3", "qwen2.5": "qwen3", "qwen2": "qwen3",
    "llama-3.3": "llama3", "llama-3.2": "llama3", "llama-3.1": "llama3",
    "llama-3": "llama3", "llama3": "llama3", "meta-llama-3": "llama3",
}


def get_encoding(name: str, data_dir: str = "") -> "Tokenizer":
    """Load (and cache) a tokenizer by encoding name.

    Bundled: 'cl100k_base', 'o200k_base', 'o200k_harmony' (GPT-OSS), 'llama3',
    'qwen3' (Qwen2.5/Qwen3). Encodings imported with quicktok.import_tokenizer()
    are found automatically. 'llama4' needs a data_dir holding llama4.vocab +
    llama4.special (gated Meta vocab — see tools/export_llama4.py)."""
    key = (name, data_dir)
    if key not in _CACHE:
        if data_dir:
            _CACHE[key] = Tokenizer(name, data_dir)
        else:
            try:
                _CACHE[key] = Tokenizer(name, "")          # bundled data
            except RuntimeError:
                # encodings imported via quicktok.import_tokenizer() live in the
                # user data dir ($QUICKTOK_DATA or ~/.cache/quicktok)
                from .importer import data_home
                _CACHE[key] = Tokenizer(name, data_home())
    return _CACHE[key]


def encoding_for_model(model: str) -> "Tokenizer":
    """tiktoken-style: resolve a model name to its encoding."""
    m = model.lower().rsplit("/", 1)[-1]    # "meta-llama/Llama-3.1-8B" -> "llama-3.1-8b"
    for prefix, enc in sorted(MODEL_TO_ENCODING.items(), key=lambda kv: -len(kv[0])):
        if m == prefix or m.startswith(prefix + "-") or m.startswith(prefix + "."):
            return get_encoding(enc)
    raise KeyError(f"unknown model {model!r}; pass an encoding name to get_encoding(), "
                   f"or import the model's tokenizer with quicktok.import_tokenizer()")


def count_batch(enc: "Tokenizer", texts, threads: int = 0):
    """Token counts for many texts, in parallel. Returns a numpy int64 array.
    Faster than len(encode(t)) per text — no Python list of ids is ever built."""
    import numpy as _np
    _toks, offsets = enc.encode_batch(list(texts), threads)
    return _np.diff(offsets)


def import_tokenizer(source, name, data_dir=None, corpus=()):
    """Import + verify a tokenizer.json / tekken.json (see quicktok.importer)."""
    from .importer import import_tokenizer as _imp
    return _imp(source, name, data_dir, corpus)


__all__ = ["Tokenizer", "get_encoding", "encoding_for_model", "count_batch",
           "import_tokenizer", "MODEL_TO_ENCODING", "__version__"]
