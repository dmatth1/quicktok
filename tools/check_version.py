#!/usr/bin/env python3
"""Assert every version string in the repo matches VERSION. Run in CI."""
import re, sys, pathlib
root = pathlib.Path(__file__).resolve().parent.parent
want = (root / "VERSION").read_text().strip()
checks = {
    "pyproject.toml": r'version = "([\d.]+)"',
    "CMakeLists.txt": r'project\(quicktok VERSION ([\d.]+)',
    "Makefile": r'VERSION\s+:= ([\d.]+)',
    "quicktok.pc.in": r'Version: ([\d.]+)',
    "python/src/_quicktok.cpp": r'__version__"\) = "([\d.]+)"',
}
bad = []
for f, pat in checks.items():
    m = re.search(pat, (root / f).read_text())
    got = m.group(1) if m else None
    print(f"  {got or '???':8} {f}")
    if got != want:
        bad.append(f"{f}: {got!r} != VERSION {want!r}")
if bad:
    print("\nVERSION MISMATCH:"); [print(" ", b) for b in bad]; sys.exit(1)
print(f"\nall versions == {want}")
