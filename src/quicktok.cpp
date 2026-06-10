#include "quicktok.hpp"
#include "bpe.hpp"
#include "pretok.hpp"

namespace quicktok {

struct Tokenizer::Impl {
    Vocab V;
    UClass U;
};

Tokenizer::Tokenizer() : impl(new Impl) {}
Tokenizer::~Tokenizer() = default;
Tokenizer::Tokenizer(Tokenizer&&) noexcept = default;
Tokenizer& Tokenizer::operator=(Tokenizer&&) noexcept = default;

Tokenizer Tokenizer::load(const std::string& vocab_path, const std::string& uniclass_path) {
    Tokenizer t;
    t.impl->V = Vocab::load(vocab_path.c_str());
    t.impl->U = UClass::load(uniclass_path.c_str());
    return t;
}
Tokenizer Tokenizer::load_dir(const std::string& dir) {
    return load(dir + "/cl100k.vocab", dir + "/uniclass.bin");
}

size_t Tokenizer::vocab_size() const { return impl->V.size(); }

// pieces starting with a 3-byte UTF-8 lead take the multibyte-optimized encoder
// (own TU; see src/bpe.hpp for why the dispatch lives at the piece level).
static inline void merge_piece(const Vocab& V, const uint8_t* piece, size_t len,
                               std::vector<uint32_t>& out) {
    if (len >= 3 && piece[0] >= 0xE0 && piece[0] < 0xF0) { V.encode_mb(piece, (uint32_t)len, out); return; }
    V.encode(piece, (uint32_t)len, out);
}

void Tokenizer::encode(const uint8_t* t, size_t len, std::vector<uint32_t>& out) const {
    const Vocab& V = impl->V;
    const UClass& U = impl->U;
    uint32_t L = (uint32_t)len, p = 0;
    // fused pretok+merge loop; ASCII-word fast path skips the alternation cascade
    // and reuses its trie walk when the word is a single token (the common case).
    while (p < L) {
        uint8_t b0 = t[p]; uint32_t ls = (b0 == ' ') ? p + 1 : p;
        if (ls < L && (uint8_t)((t[ls] | 0x20) - 'a') <= 25u) {       // ASCII word
            uint32_t we = ascii_letter_run(t, ls, L);
            if (we == L || t[we] < 0x80) {                            // pure-ASCII word
                uint32_t wlen = we - p;
                uint32_t first = V.next_match(t + p, wlen);
                if (V.token_len(first) == wlen) out.push_back(first); // single token: fused
                else V.encode_with_first(t + p, wlen, first, out);    // reuse the walk
                p = we; continue;
            }
        }
        uint32_t l = pretok_next(U, t, p, L);
        merge_piece(V, t + p, l, out); p += l;
    }
}

std::vector<uint32_t> Tokenizer::encode(std::string_view text) const {
    std::vector<uint32_t> out;
    out.reserve(text.size() / 3 + 8);
    encode((const uint8_t*)text.data(), text.size(), out);
    return out;
}

void Tokenizer::decode(const uint32_t* ids, size_t n, std::string& out) const {
    const Vocab& V = impl->V;
    for (size_t i = 0; i < n; i++) {
        uint32_t id = ids[i];
        if (id >= V.n) continue;
        out.append((const char*)V.all.data() + V.tstart[id], V.tstart[id+1] - V.tstart[id]);
    }
}
std::string Tokenizer::decode(const std::vector<uint32_t>& ids) const {
    std::string out; out.reserve(ids.size() * 4);
    decode(ids.data(), ids.size(), out);
    return out;
}

}  // namespace quicktok
