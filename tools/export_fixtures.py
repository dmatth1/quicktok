#!/usr/bin/env python3
"""Regenerate quicktok's data/ from the live reference (tiktoken + regex).
Needs: pip install tiktoken regex.

  cl100k.vocab : u32 n; n records of (u16 blen, blen bytes, u32 rank)
                 == tiktoken cl100k_base mergeable_ranks.
  uniclass.bin : \\p{L}/\\p{N}/\\s codepoint ranges (see tools/export_unicode.py),
                 byte-exact to the regex engine that matches tiktoken's split.

Run tools/export_unicode.py separately (it owns uniclass.bin + its .meta stamp).
"""
import os, struct, tiktoken

DATA = os.path.join(os.path.dirname(__file__), "..", "data")

def write_specials(stem, specials):
    # specials: u32 n; per entry u32 id, u16 len, bytes (sorted by id)
    sp = sorted(specials.items(), key=lambda kv: kv[1])
    with open(os.path.join(DATA, stem + ".special"), "wb") as f:
        f.write(struct.pack("<I", len(sp)))
        for tok, tid in sp:
            b = tok.encode()
            f.write(struct.pack("<I", tid)); f.write(struct.pack("<H", len(b))); f.write(b)
    return len(sp)

def export(name, stem, vocab=True):
    enc = tiktoken.get_encoding(name)
    if vocab:
        ranks = enc._mergeable_ranks
        with open(os.path.join(DATA, stem + ".vocab"), "wb") as f:
            f.write(struct.pack("<I", len(ranks)))
            for b, r in ranks.items():
                f.write(struct.pack("<H", len(b))); f.write(b); f.write(struct.pack("<I", r))
        n = write_specials(stem, enc._special_tokens)
        print(f"{name}: {len(ranks)} tokens + {n} specials -> data/{stem}.vocab/.special")
    else:
        n = write_specials(stem, enc._special_tokens)
        print(f"{name}: {n} specials -> data/{stem}.special (reuses an existing .vocab)")

def main():
    os.makedirs(DATA, exist_ok=True)
    export("cl100k_base", "cl100k")
    export("o200k_base", "o200k")
    # o200k_harmony (GPT-OSS) has the SAME pattern AND merge ranks as o200k_base;
    # only the special tokens differ, so it reuses data/o200k.vocab.
    export("o200k_harmony", "o200k_harmony", vocab=False)

if __name__ == "__main__":
    main()
