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

static int run_vectors(const quicktok::Tokenizer& tok, const std::string& path, bool special) {
    auto buf = rd(path);
    const uint8_t* p = buf.data(); uint32_t ncase; memcpy(&ncase, p, 4); p += 4;
    int fails = 0;
    for (uint32_t c = 0; c < ncase; c++) {
        uint32_t blen; memcpy(&blen, p, 4); p += 4;
        std::string text((const char*)p, blen); p += blen;
        uint32_t nid; memcpy(&nid, p, 4); p += 4;
        std::vector<uint32_t> ref(nid);
        if (nid) memcpy(ref.data(), p, nid * 4ull);
        p += nid * 4ull;

        auto got = special ? tok.encode_with_special(text) : tok.encode(text);
        bool enc_ok = (got == ref);
        std::string back = tok.decode(got);
        bool dec_ok = (back == text);
        if (!special && tok.count(text) != got.size()) { fails++; printf("  FAIL case %u: count mismatch\n", c); }
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
    return fails;
}

int main(int argc, char** argv) {
    std::string data = argc > 1 ? argv[1] : "data";
    int fails = 0;
    auto tok = quicktok::Tokenizer::load_dir(data);
    {
        printf("cl100k_base: vocab=%zu\n", tok.vocab_size());
        fails += run_vectors(tok, "test/vectors.bin", false);
        fails += run_vectors(tok, "test/vectors_special.bin", true);
        if (!fails) printf("cl100k_base: vectors + specials exact\n");
    }
    {
        auto t2 = quicktok::Tokenizer::load_dir(data, "o200k_base");
        printf("o200k_base: vocab=%zu\n", t2.vocab_size());
        fails += run_vectors(t2, "test/vectors_o200k.bin", false);
        fails += run_vectors(t2, "test/vectors_o200k_special.bin", true);
        if (!fails) printf("o200k_base: vectors + specials exact\n");
    }

    // --- Llama-3 ---
    {
        auto l3 = quicktok::Tokenizer::load_dir(data, "llama3");
        printf("llama3: vocab=%zu\n", l3.vocab_size());
        fails += run_vectors(l3, "test/vectors_llama3.bin", false);
        fails += run_vectors(l3, "test/vectors_llama3_special.bin", true);
        if (!fails) printf("llama3: vectors + specials exact\n");
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

    // --- encode_batch: parallel batch == sequential encode ---
    {
        std::vector<std::string> texts;
        for (int i = 0; i < 257; i++)
            texts.push_back("batch item " + std::to_string(i) + ": the quick brown fox 123 日本語");
        std::vector<std::string_view> views(texts.begin(), texts.end());
        auto batch = tok.encode_batch(views);          // default threads
        int bad = 0;
        for (size_t i = 0; i < texts.size(); i++)
            if (batch[i] != tok.encode(texts[i])) bad++;
        if (bad) { fails++; printf("  FAIL: encode_batch diverged on %d items\n", bad); }
        else printf("encode_batch: OK (257 texts, parallel == sequential)\n");
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

    // --- tight-buffer / no-OOB: encode must not read past the input (ASan catches
    //     the alt2 prefix-probe overflow the fuzzer found). Inputs whose last piece
    //     is a lone prefix-eligible non-letter are the trigger. ---
    {
        const char* edge[] = {"!", "a!", "x@", "hi?", " #", "1+", "foo*", "z\xc2", "\xe0\x80"};
        for (auto c : edge)
            for (auto& enc : {std::string("cl100k_base"), std::string("o200k_base")}) {
                auto e = quicktok::Tokenizer::load_dir(data, enc);
                size_t n = strlen(c);
                std::vector<uint8_t> tight(c, c + n);   // no trailing slack
                std::vector<uint32_t> ids;
                e.encode(tight.data(), tight.size(), ids);
            }
        printf("no-OOB: OK (tight-buffer edge inputs, both encodings)\n");
    }

    if (fails == 0) printf("PASS: both encodings exact (encode == tiktoken incl. specials, decode round-trips)\n");
    else            printf("FAIL: %d failures\n", fails);
    return fails ? 1 : 0;
}
