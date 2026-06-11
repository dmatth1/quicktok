#!/usr/bin/env python3
"""quicktok vs tiktoken — single-thread and parallel batch, on bench/corpus.txt.

Both libraries are exercised through their native APIs (quicktok.encode /
encode_batch, tiktoken.encode_ordinary / encode_ordinary_batch), so this is the
apples-to-apples comparison a Python consumer actually sees. Exactness is checked
before timing. Run: make bench-py   (needs: pip install quicktok tiktoken)
"""
import os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "corpus.txt")


def best_of(reps, fn):
    fn()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
    return best


def main():
    try:
        import quicktok, tiktoken
    except ImportError as e:
        sys.exit(f"need quicktok + tiktoken installed: {e}")

    text = open(CORPUS, encoding="utf-8").read()
    docs = [d for d in text.split("\n\n") if d]
    mb = len(text.encode()) / 1e6
    nthreads = os.cpu_count() or 8
    print(f"quicktok {quicktok.__version__} vs tiktoken {tiktoken.__version__}")
    print(f"corpus {mb:.2f} MB, {len(docs)} docs, {nthreads} cores\n")

    for enc in ("cl100k_base", "o200k_base"):
        qt = quicktok.get_encoding(enc)
        tk = tiktoken.get_encoding(enc)
        # exactness gate
        assert qt.encode(text) == tk.encode_ordinary(text), f"{enc}: id mismatch!"
        ntok = len(qt.encode(text))
        print(f"[{enc}]  {ntok} tokens, exact vs tiktoken ✓")

        q1 = best_of(5, lambda: qt.encode(text))
        t1 = best_of(5, lambda: tk.encode_ordinary(text))
        print(f"  single-thread   quicktok {mb/q1:7.1f} MB/s   tiktoken {mb/t1:6.1f} MB/s   {t1/q1:.1f}x")

        T = nthreads
        qb = best_of(5, lambda: qt.encode_batch(docs, T))
        qn = best_of(5, lambda: qt.encode_batch_numpy(docs, T))
        tb = best_of(5, lambda: tk.encode_ordinary_batch(docs, num_threads=T))
        print(f"  batch ({T:>2} thr) list    quicktok {mb/qb:7.1f} MB/s   tiktoken {mb/tb:6.1f} MB/s   {tb/qb:.1f}x")
        print(f"  batch ({T:>2} thr) numpy   quicktok {mb/qn:7.1f} MB/s   ({q1/qn:.1f}x its single-thread, {tb/qn:.0f}x tiktoken batch)\n")


if __name__ == "__main__":
    main()
