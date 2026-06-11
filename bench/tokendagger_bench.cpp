// TokenDagger (github.com/M4THYOU/TokenDagger) on one corpus file — the optional
// TokenDagger row of the README tables. C++ vs C++: benches its native
// tiktoken::CoreBPE::encode_ordinary (full pipeline), single thread, best-of-5,
// exact-checked against a reference ids file before timing.
//
// One-time setup (compare.py prints this too):
//   git clone --depth 1 https://github.com/M4THYOU/TokenDagger "$TOKENDAGGER_DIR"
//   make -C "$TOKENDAGGER_DIR/src/tiktoken" CXX=c++ \
//        CXXFLAGS="-std=c++17 -O2 -fPIC -w -I<pcre2-include>"
//
// Build + invocation are handled by bench/compare.py (set TOKENDAGGER_DIR):
//   tokendagger_bench <corpus.txt> <ref.ids> <pat-file> <vocab-file> <special-file>
// vocab/special files: u32 count; per item u32 rank, u32 len, bytes (dumped from
// tiktoken by compare.py — canonical vocab, same as every other encoder gets).
#include "tiktoken/tiktoken.hpp"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
using clk = std::chrono::steady_clock;

static std::vector<uint8_t> rd(const std::string& p) {
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", p.c_str()); exit(2); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> v(n); if (n && fread(v.data(), 1, n, f) != (size_t)n) exit(2);
    fclose(f); return v;
}

// VocabItem comes from TokenDagger's own tiktoken.hpp ({rank, token_bytes, token_string})
static std::vector<VocabItem> load_vocab(const char* path) {
    auto b = rd(path);
    const uint8_t* p = b.data();
    uint32_t n; memcpy(&n, p, 4); p += 4;
    std::vector<VocabItem> v(n);
    for (uint32_t i = 0; i < n; i++) {
        uint32_t rank, len; memcpy(&rank, p, 4); memcpy(&len, p + 4, 4); p += 8;
        v[i].rank = (int)rank; v[i].token_bytes.assign(p, p + len); p += len;
    }
    return v;
}

int main(int argc, char** argv) {
    if (argc < 6) { fprintf(stderr, "usage: tokendagger_bench <corpus> <ref.ids> <pat> <vocab> <special>\n"); return 2; }
    auto patv = rd(argv[3]);
    std::string pat((const char*)patv.data(), patv.size());
    tiktoken::CoreBPE bpe(pat, load_vocab(argv[4]), load_vocab(argv[5]));

    auto txtv = rd(argv[1]);
    std::string txt((const char*)txtv.data(), txtv.size());
    auto idf = rd(argv[2]);
    uint32_t nid; memcpy(&nid, idf.data(), 4);
    std::vector<uint32_t> ref(nid);
    memcpy(ref.data(), idf.data() + 4, (size_t)nid * 4);

    std::vector<int> out = bpe.encode_ordinary(txt);
    bool ok = out.size() == ref.size();
    if (ok) for (size_t i = 0; i < ref.size(); i++) if ((uint32_t)out[i] != ref[i]) { ok = false; break; }
    printf("  exact vs reference: %s (%zu vs %u tokens)\n", ok ? "OK" : "*** MISMATCH ***", out.size(), nid);

    double best = 1e30;
    for (int r = 0; r < 5; r++) {
        auto t0 = clk::now();
        volatile size_t n = bpe.encode_ordinary(txt).size(); (void)n;
        best = std::min(best, std::chrono::duration<double>(clk::now() - t0).count());
    }
    printf("tokendagger  %8.2f MB/s\n", txt.size() / 1e6 / best);
    return 0;
}
