"""HuggingFace `AutoTokenizer` drop-in.

The HF ecosystem (transformers, datasets, vLLM/SGLang/TGI preprocessing) goes
through `AutoTokenizer`, not tiktoken. `patch_transformers()` makes
`AutoTokenizer.from_pretrained(...)` transparently return a quicktok-backed
tokenizer **when quicktok supports the model's grammar**, and the unmodified HF
tokenizer otherwise — so existing code gets faster with one call and never
changes behaviour on an unsupported model.

    import quicktok
    quicktok.patch_transformers()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")  # now quicktok-backed
    ids = tok.encode(text)                                          # fast path

Exactness, not approximation: the wrapper only fast-paths an encode call whose
output it can reproduce *exactly* — plain text, no per-call options it doesn't
model, and special tokens only when the model adds them as a verified static
prefix/suffix. Anything else delegates to the real HF tokenizer.
"""

_PROBES = ("a", "The quick brown fox 12345")


def _capture_specials(hf):
    """If `add_special_tokens=True` wraps the ordinary ids in a *static*
    (prefix, suffix) of special ids, return it (possibly ([], [])). Return None
    if the wrap is content-dependent or can't be determined — then specials are
    never fast-pathed."""
    try:
        result = None
        for s in _PROBES:
            without = list(hf.encode(s, add_special_tokens=False))
            with_s = list(hf.encode(s, add_special_tokens=True))
            n = len(without)
            found = None
            for p in range(len(with_s) - n + 1):
                if with_s[p:p + n] == without:
                    found = (with_s[:p], with_s[p + n:])
                    break
            if found is None:
                return None
            if result is None:
                result = found
            elif found != result:          # not the same wrap for every input
                return None
        return result
    except Exception:
        return None


def _batch_encoding(ids):
    enc = {"input_ids": ids, "attention_mask": [1] * len(ids)}
    try:
        from transformers import BatchEncoding
        return BatchEncoding(enc)
    except Exception:
        return enc


class QuicktokBackedTokenizer:
    """Wraps a HF tokenizer + a quicktok encoding. Fast-paths the encode hot
    path through quicktok when the result is exactly reproducible; delegates
    everything else (decode, padding, truncation, tensors, chat templates,
    unknown attributes) to the wrapped HF tokenizer."""

    def __init__(self, hf, qt, prefix=None, suffix=None):
        self._hf = hf
        self._qt = qt
        self._prefix = prefix          # captured static specials, or None
        self._suffix = suffix

    def _can_fast_path(self, text, add_special_tokens, kw):
        if self._qt is None or not isinstance(text, str) or kw:
            return False
        if add_special_tokens and self._prefix is None:
            return False
        return True

    def _fast_ids(self, text, add_special_tokens):
        ids = self._qt.encode_ordinary(text)
        if add_special_tokens:
            return self._prefix + ids + self._suffix
        return ids

    def encode(self, text, add_special_tokens=True, **kw):
        if self._can_fast_path(text, add_special_tokens, kw):
            return self._fast_ids(text, add_special_tokens)
        return self._hf.encode(text, add_special_tokens=add_special_tokens, **kw)

    def __call__(self, text, add_special_tokens=True, **kw):
        if self._can_fast_path(text, add_special_tokens, kw):
            return _batch_encoding(self._fast_ids(text, add_special_tokens))
        return self._hf(text, add_special_tokens=add_special_tokens, **kw)

    def __getattr__(self, name):
        # only reached for attributes not on the wrapper itself -> delegate
        hf = self.__dict__.get("_hf")
        if hf is None:
            raise AttributeError(name)
        return getattr(hf, name)


def wrap_pretrained(hf, qt):
    """Return a quicktok-backed wrapper around `hf`, or `hf` unchanged when
    `qt` is None (quicktok can't support this model -> zero behaviour change)."""
    if qt is None:
        return hf
    prefix, suffix = (_capture_specials(hf) or (None, None))
    return QuicktokBackedTokenizer(hf, qt, prefix, suffix)


def _encoding_for(model_id):
    """Resolve a model id to a quicktok encoding via the known-model map (and
    any already-imported encoding). None if unsupported."""
    try:
        from . import encoding_for_model
        return encoding_for_model(model_id)
    except Exception:
        return None


_INHERITED = object()      # sentinel: from_pretrained wasn't AutoTokenizer's own attr
_ORIG_DESC = None          # original descriptor (or _INHERITED); not-None means patched


def patch_transformers():
    """Monkey-patch `transformers.AutoTokenizer.from_pretrained` so it returns a
    quicktok-backed tokenizer for supported models. Idempotent; reverse with
    `unpatch_transformers()`. Raises ImportError if transformers isn't installed."""
    global _ORIG_DESC
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "quicktok.patch_transformers() needs `transformers` installed "
            "(pip install transformers)."
        ) from e

    if _ORIG_DESC is not None:
        return                                   # already patched

    orig_call = AutoTokenizer.from_pretrained    # bound classmethod, still callable
    _ORIG_DESC = AutoTokenizer.__dict__.get("from_pretrained", _INHERITED)

    def from_pretrained(model_id, *args, **kwargs):
        hf = orig_call(model_id, *args, **kwargs)
        return wrap_pretrained(hf, _encoding_for(model_id))

    from_pretrained._quicktok_patched = True     # observable marker (identity is unreliable)
    AutoTokenizer.from_pretrained = staticmethod(from_pretrained)


def unpatch_transformers():
    """Restore the original `AutoTokenizer.from_pretrained`."""
    global _ORIG_DESC
    if _ORIG_DESC is None:
        return
    from transformers import AutoTokenizer
    if _ORIG_DESC is _INHERITED:
        del AutoTokenizer.from_pretrained        # fall back to the inherited one
    else:
        AutoTokenizer.from_pretrained = _ORIG_DESC
    _ORIG_DESC = None
