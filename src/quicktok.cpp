#include "quicktok.hpp"
#include "bpe.hpp"
#include "pretok.hpp"
#include "pretok_o200k.hpp"
#include <atomic>
#include <cstring>
#include <stdexcept>
#include <thread>

namespace quicktok {
using detail::Vocab;
using detail::UClass;
using detail::UClassO;

struct Tokenizer::Impl {
    Vocab V;
    std::string name;                 // "cl100k_base" | "o200k_base"
    bool o200k = false;
    UClass U;                         // cl100k-pattern classes (L/N/S)
    UClassO UO;                       // o200k-pattern classes (L/N/S/UPPER/LOWER)
    std::vector<std::pair<std::string, uint32_t>> specials;   // sorted by id
};

Tokenizer::Tokenizer() : impl(new Impl) {}
Tokenizer::~Tokenizer() = default;
Tokenizer::Tokenizer(Tokenizer&&) noexcept = default;
Tokenizer& Tokenizer::operator=(Tokenizer&&) noexcept = default;

static void load_specials(const std::string& path, std::vector<std::pair<std::string, uint32_t>>& out) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return;  // specials file is optional
    auto fail = [&] {
        fclose(f);
        throw std::runtime_error("quicktok: bad special-tokens file: " + path);
    };
    uint32_t n; if (fread(&n, 4, 1, f) != 1) fail();
    if (n > 1024) fail();
    for (uint32_t i = 0; i < n; i++) {
        uint32_t id; uint16_t len;
        if (fread(&id, 4, 1, f) != 1 || fread(&len, 2, 1, f) != 1) fail();
        if (len == 0 || len > 64) fail();
        std::string s(len, 0);
        if (fread(s.data(), 1, len, f) != len) fail();
        out.emplace_back(std::move(s), id);
    }
    fclose(f);
}

Tokenizer Tokenizer::load(const std::string& vocab_path, const std::string& uniclass_path) {
    Tokenizer t;
    t.impl->V = Vocab::load(vocab_path.c_str());
    t.impl->U = UClass::load(uniclass_path.c_str());
    t.impl->name = "cl100k_base";
    return t;
}

Tokenizer Tokenizer::load_dir(const std::string& dir, const std::string& encoding) {
    Tokenizer t;
    if (encoding == "cl100k_base") {
        t.impl->V = Vocab::load((dir + "/cl100k.vocab").c_str());
        t.impl->U = UClass::load((dir + "/uniclass.bin").c_str());
        load_specials(dir + "/cl100k.special", t.impl->specials);
    } else if (encoding == "o200k_base") {
        t.impl->V = Vocab::load((dir + "/o200k.vocab").c_str());
        t.impl->UO = UClassO::load((dir + "/uniclass_o200k.bin").c_str());
        t.impl->o200k = true;
        load_specials(dir + "/o200k.special", t.impl->specials);
    } else {
        throw std::runtime_error("quicktok: unknown encoding: " + encoding +
                                 " (supported: cl100k_base, o200k_base)");
    }
    t.impl->name = encoding;
    return t;
}

size_t Tokenizer::vocab_size() const { return impl->V.size(); }
const std::string& Tokenizer::encoding() const { return impl->name; }

// pieces starting with a 3-byte UTF-8 lead take the multibyte-optimized encoder
// (own TU; see src/bpe.hpp for why the dispatch lives at the piece level).
static inline void merge_piece(const Vocab& V, const uint8_t* piece, size_t len,
                               std::vector<uint32_t>& out) {
    if (len >= 3 && piece[0] >= 0xE0 && piece[0] < 0xF0) { V.encode_mb(piece, (uint32_t)len, out); return; }
    V.encode(piece, (uint32_t)len, out);
}

void Tokenizer::encode(const uint8_t* t, size_t len, std::vector<uint32_t>& out) const {
    if (len > 0xFFFFFFFFull)
        throw std::invalid_argument("quicktok: input exceeds 4 GiB per encode() call — split it");
    const Vocab& V = impl->V;
    uint32_t L = (uint32_t)len, p = 0;
    if (impl->o200k) {
        const UClassO& U = impl->UO;
        while (p < L) {
            uint32_t l = detail::pretok_next_o200k(U, t, p, L);
            if (!l) l = 1;
            merge_piece(V, t + p, l, out); p += l;
        }
        return;
    }
    // cl100k: fused pretok+merge loop; ASCII-word fast path skips the alternation
    // cascade and reuses its trie walk when the word is a single token.
    const UClass& U = impl->U;
    while (p < L) {
        uint8_t b0 = t[p]; uint32_t ls = (b0 == ' ') ? p + 1 : p;
        if (ls < L && (uint8_t)((t[ls] | 0x20) - 'a') <= 25u) {       // ASCII word
            uint32_t we = detail::ascii_letter_run(t, ls, L);
            if (we == L || t[we] < 0x80) {                            // pure-ASCII word
                uint32_t wlen = we - p;
                uint32_t first = V.next_match(t + p, wlen);
                if (V.token_len(first) == wlen) out.push_back(first); // single token: fused
                else V.encode_with_first(t + p, wlen, first, out);    // reuse the walk
                p = we; continue;
            }
        }
        uint32_t l = detail::pretok_next(U, t, p, L);
        merge_piece(V, t + p, l, out); p += l;
    }
}

std::vector<uint32_t> Tokenizer::encode(std::string_view text) const {
    std::vector<uint32_t> out;
    out.reserve(text.size() / 3 + 8);
    encode((const uint8_t*)text.data(), text.size(), out);
    return out;
}

std::vector<uint32_t> Tokenizer::encode_with_special(std::string_view text) const {
    std::vector<uint32_t> out;
    out.reserve(text.size() / 3 + 8);
    const auto& sp = impl->specials;
    size_t p = 0;
    while (p < text.size()) {
        // leftmost occurrence of any special token from p
        size_t best_pos = std::string_view::npos; size_t best_i = 0;
        for (size_t i = 0; i < sp.size(); i++) {
            size_t q = text.find(sp[i].first, p);
            if (q < best_pos) { best_pos = q; best_i = i; }
        }
        if (best_pos == std::string_view::npos) {
            encode((const uint8_t*)text.data() + p, text.size() - p, out);
            break;
        }
        if (best_pos > p)
            encode((const uint8_t*)text.data() + p, best_pos - p, out);
        out.push_back(sp[best_i].second);
        p = best_pos + sp[best_i].first.size();
    }
    return out;
}

std::vector<std::vector<uint32_t>> Tokenizer::encode_batch(const std::vector<std::string_view>& texts,
                                                           unsigned threads) const {
    std::vector<std::vector<uint32_t>> out(texts.size());
    if (texts.empty()) return out;
    unsigned hw = std::thread::hardware_concurrency();
    unsigned n = threads ? threads : (hw ? hw : 4);
    if (n > texts.size()) n = (unsigned)texts.size();
    if (n <= 1) {
        for (size_t i = 0; i < texts.size(); i++) out[i] = encode(texts[i]);
        return out;
    }
    std::atomic<size_t> next{0};
    std::vector<std::thread> ths;
    ths.reserve(n);
    for (unsigned t = 0; t < n; t++)
        ths.emplace_back([&] {
            for (size_t i; (i = next.fetch_add(1, std::memory_order_relaxed)) < texts.size(); )
                out[i] = encode(texts[i]);
        });
    for (auto& th : ths) th.join();
    return out;
}

size_t Tokenizer::count(std::string_view text) const {
    std::vector<uint32_t> out;
    out.reserve(text.size() / 3 + 8);
    encode((const uint8_t*)text.data(), text.size(), out);
    return out.size();
}

void Tokenizer::decode(const uint32_t* ids, size_t n, std::string& out) const {
    const Vocab& V = impl->V;
    for (size_t i = 0; i < n; i++) {
        uint32_t id = ids[i];
        if (id < V.n) {
            out.append((const char*)V.all.data() + V.tstart[id], V.tstart[id+1] - V.tstart[id]);
            continue;
        }
        for (const auto& [s, sid] : impl->specials)
            if (sid == id) { out.append(s); break; }
        // other out-of-range ids are skipped
    }
}
std::string Tokenizer::decode(const std::vector<uint32_t>& ids) const {
    std::string out; out.reserve(ids.size() * 4);
    decode(ids.data(), ids.size(), out);
    return out;
}

}  // namespace quicktok
