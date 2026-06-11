# Security Policy

quicktok parses untrusted text and untrusted data files. It is continuously tested
under AddressSanitizer + UndefinedBehaviorSanitizer and fuzzed (see `test/fuzz_*`).

- Loading a malformed vocab/data file throws `std::runtime_error` — it does not crash.
- Any byte sequence is accepted as input; invalid UTF-8 is handled, not rejected.

To report a vulnerability, please open a private security advisory on GitHub rather
than a public issue.
