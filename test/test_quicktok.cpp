// quicktok correctness test: encode == tiktoken reference ids (byte-exact), and
// decode(encode(x)) == x. Vectors in test/vectors.bin were produced by tiktoken
// cl100k_base (the reference IS the spec).
#include "quicktok.hpp"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static std::vector<uint8_t> rd(const std::string& p) {
    FILE* f = fopen(p.c_str(), "rb"); if (!f) { fprintf(stderr, "open %s\n", p.c_str()); exit(2); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> v(n); if (n && fread(v.data(), 1, n, f) != (size_t)n) exit(2); fclose(f); return v;
}

int main(int argc, char** argv) {
    std::string data = argc > 1 ? argv[1] : "data";
    auto tok = quicktok::Tokenizer::load_dir(data);
    printf("vocab=%zu\n", tok.vocab_size());

    auto buf = rd("test/vectors.bin");
    const uint8_t* p = buf.data(); uint32_t ncase; memcpy(&ncase, p, 4); p += 4;
    int fails = 0;
    for (uint32_t c = 0; c < ncase; c++) {
        uint32_t blen; memcpy(&blen, p, 4); p += 4;
        std::string text((const char*)p, blen); p += blen;
        uint32_t nid; memcpy(&nid, p, 4); p += 4;
        std::vector<uint32_t> ref(nid);
        if (nid) memcpy(ref.data(), p, nid * 4ull);
        p += nid * 4ull;

        auto got = tok.encode(text);
        bool enc_ok = (got == ref);
        std::string back = tok.decode(got);
        bool dec_ok = (back == text);
        if (!enc_ok || !dec_ok) {
            fails++;
            printf("  FAIL case %u (%u bytes): encode=%s decode=%s  got %zu ids, want %zu\n",
                   c, blen, enc_ok ? "ok" : "MISMATCH", dec_ok ? "ok" : "MISMATCH",
                   got.size(), ref.size());
            size_t i = 0; for (; i < got.size() && i < ref.size(); i++) if (got[i] != ref[i]) break;
            if (i < got.size() || i < ref.size())
                printf("    first diff at %zu: got %u want %u\n", i,
                       i < got.size() ? got[i] : 0u, i < ref.size() ? ref[i] : 0u);
        }
    }
    if (fails == 0) printf("PASS: all %u vectors exact (encode == tiktoken, decode round-trips)\n", ncase);
    else            printf("FAIL: %d/%u vectors\n", fails, ncase);
    return fails ? 1 : 0;
}
