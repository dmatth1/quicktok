#!/usr/bin/env python3
"""(Re)generate quicktok's Qwen data files (qwen3.vocab + qwen3.special) from a
Qwen tokenizer.json. Qwen2.5 and Qwen3 share one byte-level BPE (vocab + merges);
quicktok reproduces it byte-exactly via tiktoken-rank backtracking (verified ==
the Hugging Face tokenizer; see test/vectors_qwen3.bin).

  export_qwen.py <tokenizer.json> [out_dir=../data]
  export_qwen.py --download       [out_dir=../data]   # pull Qwen3-0.6B's (Apache-2.0)

The L/N/S Unicode table is shared with cl100k (uniclass.bin, already in data/).
Qwen's pretok is cl100k's grammar with o200k-style whitespace and single-digit
\\p{N}; the C++ side selects it via the "qwen3" encoding name.
"""
import json, os, struct, sys
from urllib.request import urlopen

QWEN3_URL = "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/tokenizer.json"


def gpt2_byte_decoder():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def load(path_or_obj):
    j = path_or_obj if isinstance(path_or_obj, dict) else json.load(open(path_or_obj))
    inv = gpt2_byte_decoder()
    ranks = {}
    for s, i in j["model"]["vocab"].items():
        try: ranks[bytes(inv[c] for c in s)] = i
        except KeyError: pass
    specials = {t["content"]: t["id"] for t in j.get("added_tokens", [])}
    return ranks, specials


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    if sys.argv[1] == "--download":
        print(f"downloading {QWEN3_URL} ...")
        src = json.loads(urlopen(QWEN3_URL, timeout=60).read())
        out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "..", "data")
    else:
        src = sys.argv[1]
        out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "..", "data")
    ranks, specials = load(src)
    os.makedirs(out, exist_ok=True)
    recs = sorted(ranks.items(), key=lambda kv: kv[1])
    with open(os.path.join(out, "qwen3.vocab"), "wb") as f:
        f.write(struct.pack("<I", len(recs)))
        for raw, r in recs:
            f.write(struct.pack("<H", len(raw))); f.write(raw); f.write(struct.pack("<I", r))
    sp = sorted(specials.items(), key=lambda kv: kv[1])
    with open(os.path.join(out, "qwen3.special"), "wb") as f:
        f.write(struct.pack("<I", len(sp)))
        for tok, tid in sp:
            b = tok.encode(); f.write(struct.pack("<I", tid)); f.write(struct.pack("<H", len(b))); f.write(b)
    print(f"wrote qwen3.vocab ({len(recs)} tokens) + qwen3.special ({len(sp)}) to {out}")
    print("Qwen uses the cl100k Unicode table (uniclass.bin) — already present.")


if __name__ == "__main__":
    main()
