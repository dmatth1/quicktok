# bench/

`corpus.txt` is ~1 MB of *Moby-Dick* by Herman Melville (public domain, via
Project Gutenberg) — a fixed, in-repo corpus so `make bench` is reproducible
offline. Throughput is data-dependent; for the full cross-encoder comparison
(bpe-openai, tiktoken-rs, TokenDagger, llama.cpp) and real-dataset numbers
(The Pile, FineWeb, C4, …) see the methodology doc linked from the main README.
