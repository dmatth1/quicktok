// quicktok on one corpus file: single-thread, best-of-5, MB/s — the number that
// goes in the README tables. Optionally exact-checks against a reference ids file
// (u32 LE count, then count u32 LE ids — written by bench/compare.py from
// tiktoken) and exits nonzero on any mismatch.
//
//   build/bench_file <corpus.txt> <encoding> [data_dir] [ref.ids]
#include "quicktok.hpp"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
using clk = std::chrono::steady_clock;

static std::vector<uint8_t> rd(const char* p) {
    FILE* f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", p); exit(2); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> v(n); if (n && fread(v.data(), 1, n, f) != (size_t)n) exit(2);
    fclose(f); return v;
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: bench_file <corpus.txt> <encoding> [data_dir] [ref.ids]\n"); return 2; }
    const char* path = argv[1];
    std::string data_dir = argc > 3 ? argv[3] : "data";
    auto tok = quicktok::Tokenizer::load_dir(data_dir, argv[2]);

    auto text = rd(path);
    double mb = text.size() / 1e6;
    std::vector<uint32_t> out; out.reserve(text.size() / 3);
    auto run = [&] { out.clear(); tok.encode(text.data(), text.size(), out); };

    run();  // warm + the ids used for the exactness check
    if (argc > 4) {
        auto ref = rd(argv[4]);
        uint32_t n; memcpy(&n, ref.data(), 4);
        bool ok = (out.size() == n) && !memcmp(out.data(), ref.data() + 4, (size_t)n * 4);
        printf("  exact vs reference: %s\n", ok ? "OK" : "*** MISMATCH ***");
        if (!ok) return 1;
    }
    double best = 1e30;
    for (int r = 0; r < 5; r++) {
        auto t0 = clk::now(); run();
        best = std::min(best, std::chrono::duration<double>(clk::now() - t0).count());
    }
    printf("quicktok %-12s %8.2f MB/s  %6.2f Mtok/s  (%zu tokens, %.1f MB)\n",
           argv[2], mb / best, out.size() / best / 1e6, out.size(), mb);
    return 0;
}
