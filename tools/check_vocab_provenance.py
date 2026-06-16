#!/usr/bin/env python3
"""Verify the committed vocab files are byte-identical to their tiktoken source.
A corrupted or hand-edited data/*.vocab is otherwise only caught if it happens
to break a token vector. Run in CI (data-provenance job)."""
import struct, sys, pathlib, tiktoken

VOCABS = [("cl100k_base", "data/cl100k.vocab"), ("o200k_base", "data/o200k.vocab")]


def vocab_bytes(name):
    ranks = tiktoken.get_encoding(name)._mergeable_ranks
    out = bytearray(struct.pack("<I", len(ranks)))
    for b, r in ranks.items():
        out += struct.pack("<H", len(b)) + b + struct.pack("<I", r)
    return bytes(out)


def main():
    root = pathlib.Path(__file__).resolve().parent.parent
    bad = 0
    for name, rel in VOCABS:
        committed = (root / rel).read_bytes()
        if committed == vocab_bytes(name):
            print(f"OK   {rel}  == tiktoken {name} ({len(committed)} bytes)")
        else:
            bad += 1
            print(f"FAIL {rel}  != tiktoken {name}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
