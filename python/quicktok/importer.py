"""Import a byte-level-BPE tokenizer into quicktok, from Python — verification
included. The packaged counterpart of tools/import_tokenizer.py:

    import quicktok
    quicktok.import_tokenizer("path/or/url/to/tokenizer.json", "myenc")
    enc = quicktok.get_encoding("myenc")          # from then on, just works

Sources: a local tokenizer.json / tekken.json, an http(s) URL, or a Hugging
Face repo id like "mistralai/Mistral-Nemo-Instruct-2407" (needs
`pip install huggingface_hub`; gated repos need its auth).

Imported encodings are written to the quicktok data dir — $QUICKTOK_DATA if
set, else ~/.cache/quicktok — which get_encoding searches automatically after
the bundled data. Verification (the point): the import encodes a stress suite
plus any `corpus` files with the reference tokenizer and with quicktok, and a
single token mismatch fails the import. Unsupported normalizers and
unrecognized pretokenizer patterns are refused, not approximated.

CLI:  python -m quicktok.importer <source> <name> [--corpus FILE...]
"""
import os
import sys

from . import _import_core as core


def data_home():
    """Where imported encodings live: $QUICKTOK_DATA or ~/.cache/quicktok."""
    d = os.environ.get("QUICKTOK_DATA")
    if not d:
        d = os.path.join(os.environ.get("XDG_CACHE_HOME",
                                        os.path.join(os.path.expanduser("~"), ".cache")),
                         "quicktok")
    os.makedirs(d, exist_ok=True)
    return d


def _resolve_source(source):
    if os.path.exists(source):
        return source
    if source.startswith(("http://", "https://")):
        import urllib.request
        dst = os.path.join(data_home(), os.path.basename(source) or "tokenizer.json")
        urllib.request.urlretrieve(source, dst)
        return dst
    if "/" in source:   # Hugging Face repo id
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            sys.exit(f"{source!r} looks like a Hugging Face repo id — "
                     f"pip install huggingface_hub (and authenticate for gated repos), "
                     f"or pass a local tokenizer.json path.")
        for fname in ("tokenizer.json", "tekken.json"):
            try:
                return hf_hub_download(source, fname)
            except Exception:
                continue
        sys.exit(f"no tokenizer.json or tekken.json found in {source!r}")
    sys.exit(f"source not found: {source!r}")


def import_tokenizer(source, name, data_dir=None, corpus=()):
    """Import + verify; returns the data dir the encoding was written to."""
    from . import Tokenizer, get_encoding  # package binding (the real engine)

    path = _resolve_source(source)
    is_tekken = "tekken" in os.path.basename(path).lower()
    ranks, specials, pattern, nfc, id_offset = \
        (core.parse_tekken if is_tekken else core.parse_hf)(path)

    scanner = core.KNOWN_PATTERNS.get(pattern)
    if scanner is None:
        sys.exit("REFUSED: unrecognized pretokenizer pattern — quicktok compiles each\n"
                 "grammar by hand (no general regex engine; that is why it is fast).\n"
                 "If this model matters, file an issue with the pattern below.\n\n"
                 f"  pattern: {pattern}\n\n"
                 "  supported grammars: cl100k, o200k (Llama-4), llama3, qwen, tekken")
    if max(ranks.values()) >= (1 << 18):
        sys.exit(f"REFUSED: max token id {max(ranks.values())} exceeds the 18-bit engine limit.")

    out = data_dir or data_home()
    bundled = os.path.join(os.path.dirname(__file__), "data")
    core.write_files(out, name, ranks, specials, scanner, nfc, id_offset)
    core.ensure_tables(out, bundled, scanner, nfc)
    print(f"imported {name}: {len(ranks)} tokens, scanner={scanner}"
          + (", nfc" if nfc else "") + (f", id_offset={id_offset}" if id_offset else "")
          + f" -> {out}")

    # ---- verification (refuse to ship a wrong encoding) ----
    if is_tekken:
        try:
            from mistral_common.tokens.tokenizers.tekken import Tekkenizer, SpecialTokenPolicy
            tk = Tekkenizer.from_file(path)
            tk.special_token_policy = SpecialTokenPolicy.IGNORE
            ref_encode = lambda s: list(tk.encode(s, bos=False, eos=False))
        except ImportError:
            try:
                import tiktoken
            except ImportError:
                sys.exit("verification needs a reference: pip install mistral-common (preferred) or tiktoken")
            enc = tiktoken.Encoding(name=name, pat_str=pattern, mergeable_ranks=ranks, special_tokens={})
            ref_encode = lambda s: [i + id_offset for i in enc.encode_ordinary(s)]
    else:
        try:
            from tokenizers import Tokenizer as HFTok
        except ImportError:
            sys.exit("verification needs the reference: pip install tokenizers")
        hf = HFTok.from_file(path)
        ref_encode = lambda s: hf.encode(s, add_special_tokens=False).ids

    qt = Tokenizer(name, out)
    texts = list(core.STRESS) + [open(p, encoding="utf-8").read() for p in corpus]
    total = bad = 0
    for s in texts:
        ref = list(ref_encode(s))
        got = qt.encode(s)
        total += len(ref)
        if got != ref:
            bad += 1
    if bad:
        os.remove(os.path.join(out, name + ".enc"))
        sys.exit(f"VERIFICATION FAILED: {bad}/{len(texts)} inputs diverged — removed {name}.enc.")
    print(f"VERIFIED: {total} tokens across {len(texts)} inputs, all byte-exact. "
          f"Use quicktok.get_encoding({name!r}).")
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m quicktok.importer", description=__doc__)
    ap.add_argument("source"); ap.add_argument("name")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--corpus", nargs="*", default=[])
    a = ap.parse_args(argv)
    import_tokenizer(a.source, a.name, a.data_dir, a.corpus)


if __name__ == "__main__":
    main()
