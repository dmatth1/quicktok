#!/usr/bin/env python3
"""(Re)generate quicktok's Llama-4 data files (llama4.vocab + llama4.special).

The Llama-4 tokenizer is GATED (Meta Llama 4 Community License) and is NOT
redistributed in this repo — unlike o200k/cl100k/Qwen, quicktok cannot ship or
self-verify it. Once you have access, point this at Meta's tokenizer file:

  export_llama4.py <tokenizer.model> [out_dir=../data]   # tiktoken-format (base64 ranks)
  export_llama4.py <tokenizer.json>  [out_dir=../data]   # HF byte-level vocab

Llama-4's pat_str is byte-identical to o200k_base, so the C++ "llama4" encoding
reuses the o200k scanner + uniclass_o200k.bin (already in data/) — you only need
to supply the vocab. After exporting, regenerate the test vectors with
  tools/gen_vectors.py --llama4 <tokenizer.model>
to get the same byte-exact guarantee the bundled encodings carry.
"""
import base64, os, struct, sys


def gpt2_byte_decoder():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def from_tiktoken_model(path):
    """Meta's tokenizer.model: lines of 'base64(token) rank'. Specials live in the
    Python tokenizer wrapper, not the file — pass them via a sidecar if needed."""
    ranks = {}
    for line in open(path, "rb"):
        line = line.strip()
        if not line:
            continue
        tok_b64, rank = line.split()
        ranks[base64.b64decode(tok_b64)] = int(rank)
    return ranks, {}


def from_tokenizer_json(path):
    import json
    j = json.load(open(path))
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
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "..", "data")
    ranks, specials = (from_tokenizer_json if src.endswith(".json") else from_tiktoken_model)(src)
    os.makedirs(out, exist_ok=True)
    recs = sorted(ranks.items(), key=lambda kv: kv[1])
    with open(os.path.join(out, "llama4.vocab"), "wb") as f:
        f.write(struct.pack("<I", len(recs)))
        for raw, r in recs:
            f.write(struct.pack("<H", len(raw))); f.write(raw); f.write(struct.pack("<I", r))
    sp = sorted(specials.items(), key=lambda kv: kv[1])
    with open(os.path.join(out, "llama4.special"), "wb") as f:
        f.write(struct.pack("<I", len(sp)))
        for tok, tid in sp:
            b = tok.encode(); f.write(struct.pack("<I", tid)); f.write(struct.pack("<H", len(b))); f.write(b)
    print(f"wrote llama4.vocab ({len(recs)} tokens) + llama4.special ({len(sp)}) to {out}")
    print("Llama-4 reuses the o200k Unicode table (uniclass_o200k.bin) — already present.")


if __name__ == "__main__":
    main()
