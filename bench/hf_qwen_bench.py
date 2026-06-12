#!/usr/bin/env python3
"""quicktok vs Hugging Face `tokenizers` on the Qwen3 encoding — the open-model
table in the main README. Needs: pip install tokenizers; corpora from
bench/fetch_corpus.py; the quicktok side builds automatically.

  python bench/hf_qwen_bench.py [pile code commoncrawl]

Method: single thread, best-of-N, MB/s. HF is timed per document (its best
case — one 25 MB string drops it under ~2.2 MB/s from per-piece offset
tracking). Exactness gate: both encoders get the RAW corpus bytes; quicktok
implements the same NFC normalization HF's pipeline runs, so the ids must match
token-for-token.
"""
import os, struct, subprocess, sys, time

BENCH = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BENCH)
CORPUS_DIR = os.path.join(BENCH, "corpus")
QWEN_JSON = os.path.join(CORPUS_DIR, "qwen3_tokenizer.json")
QWEN_URL = "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/tokenizer.json"


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from tokenizers import Tokenizer
    corpora = sys.argv[1:] or ["pile", "code", "commoncrawl"]
    missing = [c for c in corpora if not os.path.exists(os.path.join(CORPUS_DIR, c + ".txt"))]
    if missing:
        subprocess.run([sys.executable, os.path.join(BENCH, "fetch_corpus.py"), *missing], check=True)
    if not os.path.exists(QWEN_JSON):
        from urllib.request import urlretrieve
        print(f"fetching Qwen3 tokenizer.json (Apache-2.0) ...")
        urlretrieve(QWEN_URL, QWEN_JSON)
    subprocess.run(["make", "-C", ROOT, "bench-tools"], check=True, capture_output=True)
    bench_file = os.path.join(ROOT, "build", "bench_file")
    hf = Tokenizer.from_file(QWEN_JSON)

    for c in corpora:
        path = os.path.join(CORPUS_DIR, c + ".txt")
        raw = open(path, "rb").read()
        text = raw.decode("utf-8")
        mb = len(raw) / 1e6
        print(f"\n### {c} ({mb:.1f} MB, raw)")

        # reference ids (HF is the spec for qwen3; it NFC-normalizes internally)
        ids = hf.encode(text, add_special_tokens=False).ids
        ids_path = os.path.join(CORPUS_DIR, f"{c}.qwen3.ids")
        with open(ids_path, "wb") as f:
            f.write(struct.pack("<I", len(ids)))
            f.write(struct.pack(f"<{len(ids)}I", *ids))

        # quicktok on the same RAW bytes (exact-checked inside bench_file)
        out = subprocess.run([bench_file, path, "qwen3", os.path.join(ROOT, "data"), ids_path],
                             check=True, capture_output=True, text=True).stdout
        print(out.strip())

        # HF per-document, single thread, best-of-3
        docs = [d for d in text.split("\n\n") if d]
        def run():
            for d in docs:
                hf.encode(d, add_special_tokens=False)
        run()
        best = float("inf")
        for _ in range(3):
            t0 = time.perf_counter(); run(); best = min(best, time.perf_counter() - t0)
        print(f"HF tokenizers (per-doc) {mb/best:7.2f} MB/s  ({len(docs)} docs)")


if __name__ == "__main__":
    main()
