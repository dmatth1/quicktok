#!/usr/bin/env python3
"""Export/verify Unicode class membership (\\p{L}, \\p{N}, \\s) as codepoint ranges,
for the o200k_base (GPT-4o) pretokenizer, whose letter alternatives are CASE-AWARE:
  UPPER = [\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]   LOWER = [\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]
Five classes (L, N, S, UPPER, LOWER) from the SAME `regex` engine verified ==
tiktoken's split.

The pinning contract: id-exactness
always means matching some specific build's Unicode snapshot — tiktoken's own
tables are whatever its fancy-regex/regex-syntax lockfile vendored. We make that
pinning EXPLICIT (versions stamped in uniclass.bin.meta) and VERIFIABLE
(`verify` re-derives all 1,114,112 codepoints x 3 classes against the live
reference engine and diffs — a complete equivalence proof per version).

format: for each class L,N,S: u32 nranges; nranges*(u32 lo, u32 hi inclusive)

usage:
  export_unicode_o200k.py            # export -> fixtures/uniclass.bin + .meta stamp
  export_unicode.py verify     # FULL re-derivation diff vs the committed table
                               # + checksum check vs .meta; exit 1 on any mismatch
"""
import hashlib, os, struct, sys
import regex

FIX = os.path.join(os.path.dirname(__file__), "..", "data")
OUT = os.path.join(FIX, "uniclass_o200k.bin")
META = OUT + ".meta"
CLASSES = [("L", r"\p{L}"), ("N", r"\p{N}"), ("S", r"\s"),
           ("UPPER", r"[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]"),
           ("LOWER", r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]")]


def ranges_for(pat):
    rx = regex.compile(pat)
    rs = []
    cp = 0
    while cp <= 0x10FFFF:
        if 0xD800 <= cp <= 0xDFFF:  # surrogates: not valid chars
            cp = 0xE000
            continue
        try:
            m = bool(rx.match(chr(cp)))
        except Exception:
            m = False
        if m:
            lo = cp
            while cp + 1 <= 0x10FFFF:
                nxt = cp + 1
                if 0xD800 <= nxt <= 0xDFFF:
                    break
                try:
                    mm = bool(rx.match(chr(nxt)))
                except Exception:
                    mm = False
                if not mm:
                    break
                cp = nxt
            rs.append((lo, cp))
        cp += 1
    return rs


def derive_all():
    return [(name, ranges_for(pat)) for name, pat in CLASSES]


def encode(classes):
    out = bytearray()
    for _, rs in classes:
        out += struct.pack("<I", len(rs))
        for lo, hi in rs:
            out += struct.pack("<II", lo, hi)
    return bytes(out)


def decode(blob):
    classes, off = [], 0
    for name, _ in CLASSES:
        (n,) = struct.unpack_from("<I", blob, off); off += 4
        rs = []
        for _ in range(n):
            lo, hi = struct.unpack_from("<II", blob, off); off += 8
            rs.append((lo, hi))
        classes.append((name, rs))
    return classes


def stamp(blob):
    import unicodedata
    return (
        f"sha256: {hashlib.sha256(blob).hexdigest()}\n"
        f"regex-module: {regex.__version__}\n"
        f"python: {sys.version.split()[0]}\n"
        f"cpython-unicodedata-ucd: {unicodedata.unidata_version} (informative; regex bundles its own UCD)\n"
        f"classes: L N S UPPER LOWER (o200k case-aware letter grammar) over all codepoints excl. surrogates\n"
        f"reference-chain: this engine's split == tiktoken cl100k ids, verified via the\n"
        f"  byte-exact id oracle (unicode stress fixtures). Re-run `verify` after\n"
        f"  any regex/tiktoken upgrade; review the range diff before re-exporting.\n"
    )


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "export"
    if mode == "export":
        os.makedirs(FIX, exist_ok=True)
        classes = derive_all()
        blob = encode(classes)
        with open(OUT, "wb") as f:
            f.write(blob)
        with open(META, "w") as f:
            f.write(stamp(blob))
        for name, rs in classes:
            print(f"{name}: {len(rs)} ranges, {sum(hi-lo+1 for lo,hi in rs)} codepoints")
        print("wrote", OUT, "and", META)
        return 0

    if mode == "verify":
        blob = open(OUT, "rb").read()
        ok = True
        # 1. checksum vs meta stamp
        want = None
        if os.path.exists(META):
            for line in open(META):
                if line.startswith("sha256:"):
                    want = line.split()[1]
        got = hashlib.sha256(blob).hexdigest()
        if want is None:
            print("WARN: no .meta stamp found (pre-hardening table)"); ok = False
        elif want != got:
            print(f"FAIL: uniclass.bin sha256 {got} != stamped {want}"); ok = False
        else:
            print("checksum: OK")
        # 2. full re-derivation diff against the live reference engine
        committed = decode(blob)
        live = derive_all()
        for (name, crs), (_, lrs) in zip(committed, live):
            if crs == lrs:
                print(f"{name}: OK ({len(crs)} ranges, exhaustive over all codepoints)")
                continue
            ok = False
            cset, lset = set(crs), set(lrs)
            print(f"{name}: *** {len(crs)} committed vs {len(lrs)} live ranges ***")
            for r in sorted(lset - cset)[:20]:
                print(f"  + live-only   U+{r[0]:04X}..U+{r[1]:04X}")
            for r in sorted(cset - lset)[:20]:
                print(f"  - table-only  U+{r[0]:04X}..U+{r[1]:04X}")
        print("VERIFY:", "OK — table is byte-equivalent to the live reference engine" if ok
              else "MISMATCH — review diff; re-export only after confirming the reference moved intentionally")
        return 0 if ok else 1

    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
