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

# model name -> encoding (the common ones)
MODEL_TO_ENCODING = {
    "gpt-4o": "o200k_base", "gpt-4o-mini": "o200k_base", "o1": "o200k_base",
    "o3": "o200k_base", "gpt-4": "cl100k_base", "gpt-4-turbo": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base", "gpt-oss": "o200k_harmony",
    "qwen3": "qwen3", "qwen2.5": "qwen3", "qwen2": "qwen3",
}


def get_encoding(name: str, data_dir: str = "") -> "Tokenizer":
    """Load (and cache) a tokenizer by encoding name.

    Bundled: 'cl100k_base', 'o200k_base', 'o200k_harmony' (GPT-OSS), 'llama3',
    'qwen3' (Qwen2.5/Qwen3). 'llama4' needs a data_dir holding llama4.vocab +
    llama4.special (gated Meta vocab — see tools/export_llama4.py); its pattern
    reuses the bundled o200k tables."""
    key = (name, data_dir)
    if key not in _CACHE:
        _CACHE[key] = Tokenizer(name, data_dir)
    return _CACHE[key]


def encoding_for_model(model: str) -> "Tokenizer":
    """tiktoken-style: resolve a model name to its encoding."""
    for prefix, enc in sorted(MODEL_TO_ENCODING.items(), key=lambda kv: -len(kv[0])):
        if model == prefix or model.startswith(prefix + "-"):
            return get_encoding(enc)
    raise KeyError(f"unknown model {model!r}; pass an encoding name to get_encoding()")


def count_batch(enc: "Tokenizer", texts, threads: int = 0):
    """Token counts for many texts, in parallel. Returns a numpy int64 array.
    Faster than len(encode(t)) per text — no Python list of ids is ever built."""
    import numpy as _np
    _toks, offsets = enc.encode_batch(list(texts), threads)
    return _np.diff(offsets)


__all__ = ["Tokenizer", "get_encoding", "encoding_for_model", "count_batch",
           "MODEL_TO_ENCODING", "__version__"]
