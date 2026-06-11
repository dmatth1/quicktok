#!/usr/bin/env python3
"""quicktok vs tiktoken (and optionally TokenDagger) on the Llama-4 encoding —
the Llama-4 table in the main README. Llama-4's pretokenizer is byte-identical
to o200k_base; only the vocab differs, and Meta gates it, so this bench needs a
local `tokenizer.model` (tiktoken format, base64 ranks):

  python bench/llama4_bench.py /path/to/tokenizer.model [pile code commoncrawl]

If TOKENDAGGER_DIR is set (a TokenDagger clone with src/tiktoken built — it
also happens to redistribute Meta's tokenizer.model at src/tokenizer.model),
the TokenDagger row is benchmarked too, through its native CoreBPE.

Method: single thread, best-of-5, MB/s. The reference is Meta's own tokenizer
construction — `tiktoken.Encoding(o200k pattern, ranks from tokenizer.model)`,
exactly what meta-llama/llama-models does — and every encoder is verified
token-for-token against it before timing. Needs: pip install tiktoken.
"""
import base64, os, struct, subprocess, sys, time

BENCH = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BENCH)
CORPUS_DIR = os.path.join(BENCH, "corpus")
O200K_PAT = r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"""


def best_of(fn, reps=5):
    fn()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
    return best


def main():
    import tiktoken
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    model_path = sys.argv[1]
    corpora = sys.argv[2:] or ["pile", "code", "commoncrawl"]

    missing = [c for c in corpora if not os.path.exists(os.path.join(CORPUS_DIR, c + ".txt"))]
    if missing:
        subprocess.run([sys.executable, os.path.join(BENCH, "fetch_corpus.py"), *missing], check=True)
    subprocess.run([sys.executable, os.path.join(ROOT, "tools", "export_llama4.py"),
                    model_path, os.path.join(ROOT, "data")], check=True)
    subprocess.run(["make", "-C", ROOT, "bench-tools"], check=True, capture_output=True)
    bench_file = os.path.join(ROOT, "build", "bench_file")

    # Meta's construction: tiktoken over the model file's ranks + the o200k pattern
    ranks = {}
    for line in open(model_path, "rb"):
        line = line.strip()
        if line:
            tok, rank = line.split()
            ranks[base64.b64decode(tok)] = int(rank)
    enc = tiktoken.Encoding(name="llama4", pat_str=O200K_PAT, mergeable_ranks=ranks, special_tokens={})

    # optional TokenDagger row
    td_dir = os.environ.get("TOKENDAGGER_DIR")
    td_exe = None
    if td_dir:
        lib = os.path.join(td_dir, "src", "tiktoken", "libtiktoken.a")
        if os.path.exists(lib):
            td_exe = os.path.join(ROOT, "build", "tokendagger_bench")
            pcre2 = subprocess.run(["brew", "--prefix", "pcre2"], capture_output=True,
                                   text=True).stdout.strip() or "/usr"
            subprocess.run(["c++", "-O3", "-std=c++17", "-w", f"-I{td_dir}/src/tiktoken",
                            f"-I{td_dir}/src", f"-I{pcre2}/include",
                            os.path.join(BENCH, "tokendagger_bench.cpp"), lib,
                            f"-L{pcre2}/lib", "-lpcre2-8", "-o", td_exe], check=True)
            base = os.path.join(CORPUS_DIR, "td_llama4")
            with open(base + ".vocab", "wb") as f:
                f.write(struct.pack("<I", len(ranks)))
                for tok, rank in ranks.items():
                    f.write(struct.pack("<II", rank, len(tok))); f.write(tok)
            with open(base + ".special", "wb") as f:
                f.write(struct.pack("<I", 0))
            open(base + ".pat", "w").write(O200K_PAT)
        else:
            print(f"note: {lib} not built — skipping TokenDagger (see bench/tokendagger_bench.cpp)")
    else:
        print("note: TOKENDAGGER_DIR not set — skipping the TokenDagger row")

    for c in corpora:
        path = os.path.join(CORPUS_DIR, c + ".txt")
        raw = open(path, "rb").read()
        text = raw.decode("utf-8"); mb = len(raw) / 1e6
        print(f"\n### {c} ({mb:.1f} MB)")

        ids = enc.encode_ordinary(text)
        ids_path = os.path.join(CORPUS_DIR, f"{c}.llama4.ids")
        with open(ids_path, "wb") as f:
            f.write(struct.pack("<I", len(ids)))
            f.write(struct.pack(f"<{len(ids)}I", *ids))
        print(f"  reference: {len(ids)} tokens (Meta tiktoken construction)")

        out = subprocess.run([bench_file, path, "llama4", os.path.join(ROOT, "data"), ids_path],
                             check=True, capture_output=True, text=True).stdout
        print(out.strip())
        if td_exe:
            base = os.path.join(CORPUS_DIR, "td_llama4")
            out = subprocess.run([td_exe, path, ids_path, base + ".pat", base + ".vocab",
                                  base + ".special"], capture_output=True, text=True).stdout
            print(out.strip())
        print(f"tiktoken     {mb/best_of(lambda: enc.encode_ordinary(text)):8.2f} MB/s")


if __name__ == "__main__":
    main()
