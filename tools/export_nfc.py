#!/usr/bin/env python3
"""Export/verify the Unicode NFC tables (data/nfc.bin) used by encodings whose
reference pipeline normalizes to NFC before tokenizing (Qwen's tokenizer.json:
"normalizer": {"type": "NFC"}).

Derived from Python's unicodedata (the same UAX #15 algorithm the Hugging Face
pipeline implements); version-stamped in nfc.bin.meta and exhaustively
re-derivable. Four tables:

  suspicious : codepoint ranges that can make a string non-NFC — anything where
               NFC(cp) != cp (decomposes or is a singleton, e.g. U+2329), any
               non-starter (ccc > 0), any codepoint that can compose with a
               PRECEDING character (second elements of composition pairs,
               Hangul V/T jamo). A string with no suspicious codepoint is NFC
               by construction — that's the C++ fast path's contract.
  ccc        : nonzero canonical combining classes (canonical reordering).
  decomp     : full canonical decompositions (NFD), pre-expanded; Hangul
               syllables excluded (algorithmic in C++).
  comp       : canonical composition pairs (a,b)->c, exclusions already applied
               (a pair is included iff NFC(chr(a)+chr(b)) == chr(c)); Hangul
               excluded (algorithmic in C++).

  export_nfc.py            # export -> data/nfc.bin + .meta stamp
  export_nfc.py verify     # FULL re-derivation diff vs the committed table
"""
import hashlib, os, struct, sys, unicodedata

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
OUT = os.path.join(DATA, "nfc.bin")
META = OUT + ".meta"
HANGUL_S, HANGUL_S_END = 0xAC00, 0xD7A3
V_LO, V_HI, T_LO, T_HI = 0x1161, 0x1175, 0x11A8, 0x11C2


def cps():
    cp = 0
    while cp <= 0x10FFFF:
        if 0xD800 <= cp <= 0xDFFF:
            cp = 0xE000
            continue
        yield cp
        cp += 1


def derive():
    ccc, decomp, comp, susp = {}, {}, {}, set()
    second_elems = set()
    for cp in cps():
        ch = chr(cp)
        c = unicodedata.combining(ch)
        if c:
            ccc[cp] = c
        if not (HANGUL_S <= cp <= HANGUL_S_END):
            nfd = unicodedata.normalize("NFD", ch)
            if nfd != ch:
                decomp[cp] = [ord(x) for x in nfd]
            # single-level canonical decomposition defines composition pairs
            d = unicodedata.decomposition(ch)
            if d and not d.startswith("<"):
                parts = [int(x, 16) for x in d.split()]
                if len(parts) == 2:
                    a, b = parts
                    if unicodedata.normalize("NFC", chr(a) + chr(b)) == ch:
                        comp[(a, b)] = cp
                        second_elems.add(b)
    for cp in cps():
        ch = chr(cp)
        if (unicodedata.normalize("NFC", ch) != ch or cp in ccc
                or cp in second_elems
                or V_LO <= cp <= V_HI or T_LO <= cp <= T_HI):
            susp.add(cp)

    # suspicious set -> sorted ranges
    ranges = []
    for cp in sorted(susp):
        if ranges and cp == ranges[-1][1] + 1:
            ranges[-1][1] = cp
        else:
            ranges.append([cp, cp])
    return ranges, ccc, decomp, comp


def serialize(ranges, ccc, decomp, comp):
    out = bytearray()
    out += struct.pack("<I", len(ranges))
    for lo, hi in ranges:
        out += struct.pack("<II", lo, hi)
    out += struct.pack("<I", len(ccc))
    for cp in sorted(ccc):
        out += struct.pack("<II", cp, ccc[cp])
    out += struct.pack("<I", len(decomp))
    for cp in sorted(decomp):
        d = decomp[cp]
        out += struct.pack("<II", cp, len(d))
        out += struct.pack(f"<{len(d)}I", *d)
    out += struct.pack("<I", len(comp))
    for (a, b) in sorted(comp):
        out += struct.pack("<III", a, b, comp[(a, b)])
    return bytes(out)


def stamp(blob, ranges, ccc, decomp, comp):
    uver = unicodedata.unidata_version
    return (
        f"sha256: {hashlib.sha256(blob).hexdigest()}\n"
        f"python: {sys.version.split()[0]}\n"
        f"cpython-unicodedata-ucd: {uver}\n"
        f"tables: suspicious={len(ranges)} ranges, ccc={len(ccc)}, "
        f"decomp={len(decomp)}, comp-pairs={len(comp)}\n"
        f"contract: a string containing no suspicious codepoint is already NFC;\n"
        f"  the C++ fast path relies on this. Hangul (AC00-D7A3, jamo) handled\n"
        f"  algorithmically. Re-run `verify` after any Python/UCD upgrade.\n"
    )


def main():
    blob = serialize(*(t := derive()))
    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        committed = open(OUT, "rb").read()
        ok = committed == blob
        meta_sha = next((l.split()[1] for l in open(META)) , "")
        print(f"re-derivation: {'OK' if ok else '*** DIFFERS ***'} "
              f"({len(blob)} bytes, UCD {unicodedata.unidata_version})")
        print(f"checksum vs meta: "
              f"{'OK' if hashlib.sha256(committed).hexdigest() == meta_sha else '*** MISMATCH ***'}")
        sys.exit(0 if ok else 1)
    os.makedirs(DATA, exist_ok=True)
    open(OUT, "wb").write(blob)
    open(META, "w").write(stamp(blob, *t))
    print(f"wrote {OUT} ({len(blob)} bytes) + .meta  "
          f"(suspicious={len(t[0])} ranges, ccc={len(t[1])}, decomp={len(t[2])}, comp={len(t[3])})")


if __name__ == "__main__":
    main()
