// quicktok native benchmark — single-thread throughput + parallel batch scaling.
// Self-contained: uses bench/corpus.txt (public-domain) and data/. Reports the
// numbers; no extrapolation. For cross-encoder comparisons see the main README.
//
//   make bench   (builds + runs this)
#include "quicktok.hpp"
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <algorithm>
#include <vector>
using clk = std::chrono::steady_clock;

static std::string rd(const char* p) {
    FILE* f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", p); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::string s(n, 0); if (n && fread(&s[0], 1, n, f) != (size_t)n) exit(1); fclose(f); return s;
}

template <class F> static double best_of(int reps, F&& f) {
    f();  // warm
    double best = 1e30;
    for (int r = 0; r < reps; r++) { auto t0 = clk::now(); f(); best = std::min(best, std::chrono::duration<double>(clk::now() - t0).count()); }
    return best;
}

int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : "data";
    std::string corpus = argc > 2 ? argv[2] : "bench/corpus.txt";
    std::string text = rd(corpus.c_str());
    double MB = text.size() / 1e6;
    unsigned hw = std::thread::hardware_concurrency();
    printf("quicktok native bench — corpus %.2f MB, %u hardware threads\n\n", MB, hw);

    // split into documents (paragraphs) for the batch test
    std::vector<std::string> docs; std::vector<std::string_view> views;
    { size_t i = 0; while (i < text.size()) {
        size_t j = text.find("\n\n", i); if (j == std::string::npos) j = text.size();
        if (j > i) docs.emplace_back(text.substr(i, j - i)); i = j + 2; } }
    for (auto& d : docs) views.emplace_back(d);

    for (const char* enc : {"cl100k_base", "o200k_base"}) {
        auto tok = quicktok::Tokenizer::load_dir(dir, enc);
        std::vector<uint32_t> out; out.reserve(text.size() / 3);
        double st = best_of(7, [&]{ out.clear(); tok.encode((const uint8_t*)text.data(), text.size(), out); });
        size_t ntok = out.size();
        printf("[%s]  vocab %zu, %zu docs, %zu tokens\n", enc, tok.vocab_size(), docs.size(), ntok);
        printf("  single-thread : %7.1f MB/s   %6.2f Mtok/s\n", MB / st, ntok / st / 1e6);

        printf("  batch (encode_batch):\n");
        // thread ladder: powers of two, the machine's core count, and 2x cores to
        // show where it saturates / oversubscribes. Deduped + sorted.
        std::vector<unsigned> ladder = {1, 2, 4, 8, hw ? hw : 8u, hw ? 2u * hw : 16u};
        std::sort(ladder.begin(), ladder.end());
        ladder.erase(std::unique(ladder.begin(), ladder.end()), ladder.end());
        double t1 = 0;
        for (unsigned T : ladder) {
            double t = best_of(5, [&]{ auto r = tok.encode_batch(views, T); (void)r; });
            if (T == 1) t1 = t;
            printf("    %2u thread%-2s: %7.1f MB/s   %6.2f Mtok/s   %4.2fx\n",
                   T, T == 1 ? "" : "s", MB / t, ntok / t / 1e6, t1 / t);
        }
        printf("\n");
    }
    return 0;
}
