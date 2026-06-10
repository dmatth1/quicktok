// quicktok correctness test: encode == tiktoken reference ids (byte-exact), and
// decode(encode(x)) == x. Vectors in test/vectors.bin were produced by tiktoken
// cl100k_base (the reference IS the spec).
#include "quicktok.hpp"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
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
    // --- error handling: loads must THROW (never exit/crash) on bad inputs ---
    {
        bool threw = false;
        try { auto bad = quicktok::Tokenizer::load_dir("/nonexistent-quicktok-dir"); }
        catch (const std::exception& e) { threw = true; }
        if (!threw) { fails++; printf("  FAIL: load_dir(nonexistent) did not throw\n"); }

        // truncated vocab file
        FILE* f = fopen("/tmp/qt_trunc.vocab", "wb");
        uint32_t n = 100256; fwrite(&n, 4, 1, f); fclose(f);   // header only, no records
        threw = false;
        try { auto bad = quicktok::Tokenizer::load("/tmp/qt_trunc.vocab", data + "/uniclass.bin"); }
        catch (const std::exception& e) { threw = true; }
        if (!threw) { fails++; printf("  FAIL: truncated vocab did not throw\n"); }
        remove("/tmp/qt_trunc.vocab");
        if (fails == 0) printf("error handling: OK (bad loads throw)\n");
    }

    // --- concurrency: parallel encode() on ONE Tokenizer must stay exact ---
    {
        std::string text;
        for (int i = 0; i < 200; i++)
            text += "The quick brown fox can't jump 1234 times over donde está the lazy 犬!\n";
        auto expect = tok.encode(text);
        int bad = 0;
        std::vector<std::thread> ths;
        for (int t = 0; t < 8; t++)
            ths.emplace_back([&]{
                for (int r = 0; r < 50; r++)
                    if (tok.encode(text) != expect) __atomic_fetch_add(&bad, 1, __ATOMIC_RELAXED);
            });
        for (auto& th : ths) th.join();
        if (bad) { fails++; printf("  FAIL: concurrent encode diverged (%d)\n", bad); }
        else printf("concurrency: OK (8 threads x 50 encodes, one shared Tokenizer, all exact)\n");
    }

    if (fails == 0) printf("PASS: all %u vectors exact (encode == tiktoken, decode round-trips)\n", ncase);
    else            printf("FAIL: %d failures\n", fails);
    return fails ? 1 : 0;
}
