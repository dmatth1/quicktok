#!/usr/bin/env python3
"""Reproduce the README benchmark tables: every encoder, same method, same corpora.

  python bench/compare.py                      # full sweep -> markdown tables
  python bench/compare.py --corpora pile       # subset
  python bench/compare.py --encodings cl100k   # subset

Method (identical for every encoder): single thread, best-of-5 wall-clock over the
whole file, MB/s = bytes/best. tiktoken (the reference implementation) defines the
expected ids; every encoder is exact-checked against them before it is timed, and
a mismatch fails the run. Each comparator is called through the same raw API its
own benchmark uses.

Rows and what they need:
  quicktok (native)   builds automatically (make bench-tools)
  quicktok (Python)   pip install quicktok-v1   (skipped if missing)
  tiktoken (Python)   pip install tiktoken      (required — it's the reference)
  bpe-openai,
  tiktoken-rs         a Rust toolchain; built on first run via bench/comparators/
                      (skipped if cargo is missing)
  TokenDagger         set TOKENDAGGER_DIR to a clone with src/tiktoken built —
                      see bench/tokendagger_bench.cpp (skipped otherwise)

Corpora are fetched once into bench/corpus/ (see bench/fetch_corpus.py).
"""
import argparse, os, shutil, struct, subprocess, sys, time

BENCH = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BENCH)
CORPUS_DIR = os.path.join(BENCH, "corpus")
ENC_NAME = {"cl100k": "cl100k_base", "o200k": "o200k_base"}


def sh(cmd, **kw):
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kw)


def mbps(text_out):
    import re
    m = re.search(r"([0-9.]+) MB/s", text_out)
    return float(m.group(1)) if m else None


def ensure_corpora(names):
    missing = [n for n in names if not os.path.exists(os.path.join(CORPUS_DIR, n + ".txt"))]
    if missing:
        print(f"fetching corpora: {missing} (25 MB each)", flush=True)
        subprocess.run([sys.executable, os.path.join(BENCH, "fetch_corpus.py"), *missing], check=True)


def ensure_native():
    sh(["make", "-C", ROOT, "bench-tools"])
    return os.path.join(ROOT, "build", "bench_file")


def best_of(fn, reps=5):
    fn()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
    return best


def write_ids(path, ids):
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(ids)))
        f.write(struct.pack(f"<{len(ids)}I", *ids))


def dump_td_vocab(enc, tiktoken):
    """tiktoken's canonical vocab/pattern/specials in TokenDagger's load format."""
    e = tiktoken.get_encoding(ENC_NAME[enc])
    def dump(p, items):
        with open(p, "wb") as f:
            f.write(struct.pack("<I", len(items)))
            for tok, rank in items:
                f.write(struct.pack("<II", rank, len(tok))); f.write(tok)
    base = os.path.join(CORPUS_DIR, f"td_{enc}")
    dump(base + ".vocab", list(e._mergeable_ranks.items()))
    dump(base + ".special", [(s.encode(), r) for s, r in e._special_tokens.items()])
    open(base + ".pat", "w").write(e._pat_str)
    return base


def build_tokendagger(td_dir):
    lib = os.path.join(td_dir, "src", "tiktoken", "libtiktoken.a")
    if not os.path.exists(lib):
        print(f"  TokenDagger: {lib} not built — see bench/tokendagger_bench.cpp header; skipping")
        return None
    exe = os.path.join(ROOT, "build", "tokendagger_bench")
    pcre2 = subprocess.run(["brew", "--prefix", "pcre2"], capture_output=True, text=True).stdout.strip() or "/usr"
    try:
        sh(["c++", "-O3", "-std=c++17", "-w", f"-I{td_dir}/src/tiktoken", f"-I{td_dir}/src",
            f"-I{pcre2}/include", os.path.join(BENCH, "tokendagger_bench.cpp"), lib,
            f"-L{pcre2}/lib", "-lpcre2-8", "-o", exe])
    except subprocess.CalledProcessError as e:
        print(f"  TokenDagger build failed; skipping\n{e.stderr[-400:]}")
        return None
    return exe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", nargs="+", default=["pile", "code", "commoncrawl"])
    ap.add_argument("--encodings", nargs="+", default=["cl100k", "o200k"], choices=["cl100k", "o200k"])
    args = ap.parse_args()

    try:
        import tiktoken
    except ImportError:
        sys.exit("tiktoken is the reference and is required: pip install tiktoken")
    try:
        import quicktok as qtpy
    except ImportError:
        qtpy = None
        print("note: quicktok wheel not installed — skipping the Python-binding row")

    ensure_corpora(args.corpora)
    native = ensure_native()
    cargo = shutil.which("cargo")
    if not cargo:
        print("note: cargo not found — skipping bpe-openai / tiktoken-rs rows")
    td_dir = os.environ.get("TOKENDAGGER_DIR")
    td_exe = build_tokendagger(td_dir) if td_dir else None
    if not td_dir:
        print("note: TOKENDAGGER_DIR not set — skipping the TokenDagger row")

    results = {}  # (enc, corpus, row) -> MB/s

    for enc in args.encodings:
        for corpus in args.corpora:
            path = os.path.join(CORPUS_DIR, corpus + ".txt")
            # read as BYTES then decode: text-mode open() would translate \r\n -> \n
            # and silently hand tiktoken different bytes than the C++/Rust encoders see
            raw = open(path, "rb").read()
            text = raw.decode("utf-8")   # corpora are valid UTF-8 by construction
            mb = len(raw) / 1e6
            print(f"\n### {corpus} / {enc} ({mb:.1f} MB)")

            # 1. tiktoken — the reference: ids + its own timing
            te = tiktoken.get_encoding(ENC_NAME[enc])
            ref = te.encode_ordinary(text)
            ids_path = os.path.join(CORPUS_DIR, f"{corpus}.{enc}.ids")
            write_ids(ids_path, ref)
            results[(enc, corpus, "tiktoken (Python)")] = mb / best_of(lambda: te.encode_ordinary(text))
            print(f"  reference: {len(ref)} tokens (tiktoken)")

            # 2. quicktok native (exact-checked inside bench_file; nonzero exit on mismatch)
            out = sh([native, path, ENC_NAME[enc], os.path.join(ROOT, "data"), ids_path]).stdout
            print(out.strip())
            results[(enc, corpus, "quicktok (native)")] = mbps(out)

            # 3. quicktok Python wheel
            if qtpy:
                qe = qtpy.get_encoding(ENC_NAME[enc])
                got = qe.encode_ordinary(text)
                assert got == ref, f"quicktok(Python) MISMATCH on {corpus}/{enc}"
                results[(enc, corpus, "quicktok (Python)")] = mb / best_of(lambda: qe.encode_ordinary(text))

            # 4. bpe-openai + tiktoken-rs (exact-checked inside the crate)
            if cargo:
                out = sh([cargo, "run", "--release", "--quiet",
                          "--manifest-path", os.path.join(BENCH, "comparators", "Cargo.toml"),
                          "--", path, enc, ids_path]).stdout
                print(out.strip())
                for row in ("bpe-openai", "tiktoken-rs"):
                    for line in out.splitlines():
                        if line.startswith(row):
                            results[(enc, corpus, row)] = mbps(line)

            # 5. TokenDagger (exact-checked inside the bench)
            if td_exe:
                base = dump_td_vocab(enc, tiktoken)
                out = subprocess.run([td_exe, path, ids_path, base + ".pat", base + ".vocab",
                                      base + ".special"], capture_output=True, text=True).stdout
                print(out.strip())
                results[(enc, corpus, "TokenDagger")] = mbps(out)

    # markdown tables, same layout as the README
    rows = ["quicktok (native)", "quicktok (Python)", "bpe-openai", "tiktoken-rs",
            "tiktoken (Python)", "TokenDagger"]
    title = {"cl100k": "cl100k_base (GPT-3.5 / GPT-4)", "o200k": "o200k_base (GPT-4o)"}
    label = {"pile": "The Pile", "code": "Code", "commoncrawl": "Common Crawl"}
    print("\n" + "=" * 60)
    for enc in args.encodings:
        print(f"\n**{title[enc]}**\n")
        print("| encoder | " + " | ".join(label.get(c, c) for c in args.corpora) + " |")
        print("|---|" + "---:|" * len(args.corpora))
        for row in rows:
            vals = [results.get((enc, c, row)) for c in args.corpora]
            if all(v is None for v in vals):
                continue
            print(f"| {row} | " + " | ".join(f"{v:.1f}" if v else "—" for v in vals) + " |")
    print("\nAll timed outputs were verified token-for-token against tiktoken first.")


if __name__ == "__main__":
    main()
