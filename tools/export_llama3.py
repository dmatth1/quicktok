#!/usr/bin/env python3
"""(Re)generate quicktok's Llama-3 data files. The repo already ships data/llama3.vocab
(Meta Llama 3 Community License — see NOTICE); use this to rebuild or update it from

  * a Hugging Face tokenizer.json (e.g. an ungated mirror), or
  * a llama.cpp BPE vocab GGUF (models/ggml-vocab-llama-bpe.gguf).

  export_llama3.py tokenizer.json [out_dir=../data]
  export_llama3.py vocab.gguf     [out_dir=../data]

Writes <out_dir>/llama3.vocab and llama3.special. The L/N/S Unicode table is
shared with cl100k (uniclass.bin already in data/). Llama-3 uses tiktoken-rank
BPE; quicktok reproduces that exactly. (HF's merge-list inference can differ on
rare non-Latin+symbol sequences — see the README caveat.)
"""
import os, struct, sys


def gpt2_byte_decoder():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


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


def from_gguf(path):
    import gguf
    r = gguf.GGUFReader(path)
    f = {x.name: x for x in r.fields.values()}
    toks = f["tokenizer.ggml.tokens"]; types = f.get("tokenizer.ggml.token_type")
    inv = gpt2_byte_decoder(); ranks = {}; specials = {}
    for i, di in enumerate(toks.data):
        tt = int(types.parts[types.data[i]][0]) if types else 1
        s = bytes(toks.parts[di]).decode("utf-8")
        if tt == 1:
            try: ranks[bytes(inv[c] for c in s)] = i
            except KeyError: pass
        else:
            specials[s] = i
    return ranks, specials


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "..", "data")
    ranks, specials = (from_gguf if src.endswith(".gguf") else from_tokenizer_json)(src)
    os.makedirs(out, exist_ok=True)
    recs = sorted(ranks.items(), key=lambda kv: kv[1])
    with open(os.path.join(out, "llama3.vocab"), "wb") as f:
        f.write(struct.pack("<I", len(recs)))
        for raw, r in recs:
            f.write(struct.pack("<H", len(raw))); f.write(raw); f.write(struct.pack("<I", r))
    sp = sorted(specials.items(), key=lambda kv: kv[1])
    with open(os.path.join(out, "llama3.special"), "wb") as f:
        f.write(struct.pack("<I", len(sp)))
        for tok, tid in sp:
            b = tok.encode(); f.write(struct.pack("<I", tid)); f.write(struct.pack("<H", len(b))); f.write(b)
    print(f"wrote llama3.vocab ({len(recs)} tokens) + llama3.special ({len(sp)}) to {out}")
    print("Llama-3 uses the cl100k Unicode table (uniclass.bin) — already present.")


if __name__ == "__main__":
    main()
