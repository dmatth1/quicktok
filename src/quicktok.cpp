#include "quicktok.hpp"
#include "bpe.hpp"
#include "pretok.hpp"
#include "pretok_o200k.hpp"
#include "nfc.hpp"
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>

namespace quicktok {
using detail::Vocab;
using detail::UClass;
using detail::UClassO;
using detail::NFC;

struct Tokenizer::Impl {
    Vocab V;
    std::string name;                 // cl100k_base | o200k_base | o200k_harmony | llama3 | llama4 | qwen3
    bool o200k = false;               // use the o200k pretok scanner (o200k_base, o200k_harmony, Llama-4)
    bool llama3 = false;              // cl100k grammar + o200k-style whitespace
    bool qwen = false;                // Llama-3 whitespace + single-digit \p{N} (Qwen2.5/Qwen3)
    bool tekken = false;              // o200k grammar, no contractions, single-digit (Mistral Tekken)
    bool nfc = false;                 // reference pipeline NFC-normalizes first (Qwen)
    uint32_t id_offset = 0;           // model ids = vocab rank + offset (Tekken reserves 1000 specials)
    UClass U;                         // cl100k-pattern classes (L/N/S)
    UClassO UO;                       // o200k-pattern classes (L/N/S/UPPER/LOWER)
    NFC nfcT;                         // loaded iff nfc
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
    if (n > 4096) fail();   // o200k_harmony ships 1091 specials (200000-range reserved block)
    for (uint32_t i = 0; i < n; i++) {
        uint32_t id = 0; uint16_t len = 0;
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
    } else if (encoding == "o200k_harmony") {
        // GPT-OSS harmony: identical pattern AND merge ranks to o200k_base (so it
        // reuses o200k.vocab + the o200k scanner) — only the special tokens differ.
        t.impl->V = Vocab::load((dir + "/o200k.vocab").c_str());
        t.impl->UO = UClassO::load((dir + "/uniclass_o200k.bin").c_str());
        t.impl->o200k = true;
        load_specials(dir + "/o200k_harmony.special", t.impl->specials);
    } else if (encoding == "llama4") {
        // Llama-4: pat_str is byte-identical to o200k_base, so it reuses the o200k
        // scanner; only the trained vocab differs. The vocab is gated (Meta Llama 4
        // Community License) and not redistributed here — supply llama4.vocab via
        // tools/export_llama4.py. See README.
        t.impl->V = Vocab::load((dir + "/llama4.vocab").c_str());
        t.impl->UO = UClassO::load((dir + "/uniclass_o200k.bin").c_str());
        t.impl->o200k = true;
        load_specials(dir + "/llama4.special", t.impl->specials);
    } else if (encoding == "llama3") {
        // Llama-3: cl100k letter/number/punct grammar + o200k-style whitespace,
        // so it reuses the cl100k L/N/S table (uniclass.bin).
        t.impl->V = Vocab::load((dir + "/llama3.vocab").c_str());
        t.impl->U = UClass::load((dir + "/uniclass.bin").c_str());
        t.impl->llama3 = true;
        load_specials(dir + "/llama3.special", t.impl->specials);
    } else if (encoding == "qwen3" || encoding == "qwen2.5" || encoding == "qwen") {
        // Qwen2.5/Qwen3 share one byte-BPE: cl100k letters/contractions/punct +
        // o200k-style whitespace + single-digit \p{N}. Same L/N/S table as cl100k.
        // tiktoken-rank backtracking reproduces its merge-list output byte-exactly.
        t.impl->V = Vocab::load((dir + "/qwen3.vocab").c_str());
        t.impl->U = UClass::load((dir + "/uniclass.bin").c_str());
        t.impl->qwen = true;
        // Qwen's reference pipeline NFC-normalizes before tokenizing
        // ("normalizer": {"type": "NFC"} in its tokenizer.json)
        t.impl->nfcT = NFC::load((dir + "/nfc.bin").c_str());
        t.impl->nfc = true;
        load_specials(dir + "/qwen3.special", t.impl->specials);
    } else {
        // Data-driven encoding: <dir>/<name>.enc, written by tools/import_tokenizer.py
        // after its exactness verification passes. key=value lines:
        //   scanner=cl100k|o200k|llama3|qwen   (which hand-compiled pretok grammar)
        //   nfc=1                              (reference pipeline NFC-normalizes)
        // Vocab/specials are <name>.vocab / <name>.special; the uniclass table is
        // implied by the scanner.
        std::string encpath = dir + "/" + encoding + ".enc";
        FILE* ef = fopen(encpath.c_str(), "rb");
        if (!ef)
            throw std::runtime_error("quicktok: unknown encoding: " + encoding +
                " (built-in: cl100k_base, o200k_base, o200k_harmony, llama3, llama4, qwen3;"
                " or import one: quicktok.import_tokenizer() from Python,"
                " tools/import_tokenizer.py from a checkout)");
        char line[256];
        std::string scanner;
        bool nfc = false;
        while (fgets(line, sizeof line, ef)) {
            std::string s(line);
            while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
            if (s.empty() || s[0] == '#') continue;
            if (s.rfind("scanner=", 0) == 0) scanner = s.substr(8);
            else if (s.rfind("nfc=", 0) == 0) nfc = (s.substr(4) == "1");
            else if (s.rfind("id_offset=", 0) == 0) t.impl->id_offset = (uint32_t)strtoul(s.c_str() + 10, nullptr, 10);
            else { fclose(ef); throw std::runtime_error("quicktok: bad line in " + encpath + ": " + s); }
        }
        fclose(ef);
        t.impl->V = Vocab::load((dir + "/" + encoding + ".vocab").c_str());
        if (scanner == "o200k" || scanner == "tekken") {
            t.impl->UO = UClassO::load((dir + "/uniclass_o200k.bin").c_str());
            t.impl->o200k = true;
            if (scanner == "tekken") t.impl->tekken = true;
        } else if (scanner == "cl100k" || scanner == "llama3" || scanner == "qwen") {
            t.impl->U = UClass::load((dir + "/uniclass.bin").c_str());
            if (scanner == "llama3") t.impl->llama3 = true;
            if (scanner == "qwen") t.impl->qwen = true;
        } else {
            throw std::runtime_error("quicktok: bad scanner in " + encpath + ": '" + scanner +
                                     "' (cl100k | o200k | llama3 | qwen | tekken)");
        }
        if (nfc) {
            t.impl->nfcT = NFC::load((dir + "/nfc.bin").c_str());
            t.impl->nfc = true;
        }
        load_specials(dir + "/" + encoding + ".special", t.impl->specials);
    }
    t.impl->name = encoding;
    return t;
}

size_t Tokenizer::vocab_size() const { return impl->V.size(); }

size_t Tokenizer::n_vocab() const {
    // tiktoken semantics: max token id + 1, specials included (cl100k -> 100277)
    size_t hi = impl->id_offset + impl->V.size();          // base ids end here
    for (const auto& [s, sid] : impl->specials)
        if ((size_t)sid + 1 > hi) hi = (size_t)sid + 1;
    return hi;
}

const std::vector<std::pair<std::string, uint32_t>>& Tokenizer::special_tokens() const {
    return impl->specials;
}

bool Tokenizer::known_id(uint32_t id) const {
    if (id >= impl->id_offset && id - impl->id_offset < impl->V.n) return true;
    for (const auto& [s, sid] : impl->specials)
        if (sid == id) return true;
    return false;
}
const std::string& Tokenizer::encoding() const { return impl->name; }

static inline bool is_utf8_cont(uint8_t b) { return (b & 0xC0) == 0x80; }

// pieces starting with a *valid* 3-byte UTF-8 char take the multibyte-optimized
// encoder (own TU; see src/bpe.hpp for why the dispatch lives at the piece level).
// The mb path's r3 bootstrap masks the continuation bytes (assumes well-formed
// UTF-8), so an ill-formed 3-byte-lead sequence must fall through to the byte-
// accurate regular path — otherwise encode would be lossy on invalid UTF-8.
static inline void merge_piece(const Vocab& V, const uint8_t* piece, size_t len,
                               std::vector<uint32_t>& out) {
    if (len >= 3 && piece[0] >= 0xE0 && piece[0] < 0xF0
        && is_utf8_cont(piece[1]) && is_utf8_cont(piece[2])) {
        V.encode_mb(piece, (uint32_t)len, out); return;
    }
    V.encode(piece, (uint32_t)len, out);
}

// ASCII class predicates for the product machines below. a_pun deliberately
// includes control bytes and DEL: the regex class is [^\s\p{L}\p{N}].
static inline bool a_let(uint8_t b){ return (uint8_t)((b|0x20)-'a') <= 25u; }
static inline bool a_dig(uint8_t b){ return (uint8_t)(b-'0') <= 9u; }
static inline bool a_ws (uint8_t b){ return (uint8_t)(b-9) <= 4u || b==' '; }
static inline bool a_pun(uint8_t b){ return b < 0x80 && !a_ws(b) && !a_let(b) && !a_dig(b); }

// Single-pass product machines: one loop owns BOTH the pretok boundary rules and
// token emission for the full ASCII grammar of each family — no per-piece scanner
// dispatch on the hot path. ANY Unicode
// contact at a run boundary falls back to the exact scalar scanner for one piece,
// so output is byte-exact by construction (verified: full vector + special suites,
// 3x25MB corpus diffs vs tiktoken on both families). Tekken keeps the plain
// fused loop: its grammar is importer-verified but has no product machine yet.
static void encode_core(const Vocab& V, const UClass& Uc, const UClassO& Uo,
                        bool o200k, bool l3, bool qw, bool tk,
                        const uint8_t* t, uint32_t L, std::vector<uint32_t>& out) {
    uint32_t p = 0;
    auto emit = [&](uint32_t off, uint32_t plen) {
        V.encode_with_first(t + off, plen, V.next_match(t + off, plen), out);
    };
    auto emit_word = emit;   // whole-piece hash measured NEGATIVE at 25MB-corpus scale
                             // (L2 displacement; see evolve TOKENIZER_LOG 2026-06-12) — walk only
    if (o200k) {
        const UClassO& U = Uo;
        if (tk) {   // Tekken: plain fused loop (no product machine yet)
            while (p < L) {
                uint8_t b0 = t[p]; uint32_t ls = (b0 == ' ') ? p + 1 : p;
                if (ls < L && (uint8_t)((t[ls] | 0x20) - 'a') <= 25u) {
                    uint32_t ue = detail::ascii_upper_run(t, ls, L);
                    uint32_t we = detail::ascii_lower_run(t, ue, L);
                    if (we == L || t[we] < 0x80) { emit_word(p, we - p); p = we; continue; }
                }
                uint32_t l = detail::pretok_next_tekken(U, t, p, L);
                if (!l) l = 1;
                merge_piece(V, t + p, l, out); p += l;
            }
            return;
        }
        // o200k_base / o200k_harmony / Llama-4 product machine. Grammar mirrored
        // from pretok_next_o200k: prefix-consumed-FIRST on both word alts;
        // UPPER*LOWER+ then UPPER+LOWER* (for ASCII a pure [A-Z]* run has no
        // LOWER-eligible char, so the CJK backtrack scan never applies natively);
        // ATTACHED contractions; '/' in the punct tail; lastnl-first ws order.
        while (p < L) {
            const uint8_t b0 = t[p];
            bool fb = (b0 >= 0x80);
            uint32_t adv = 0;
            if (!fb) do {
                bool hitmb = false;
                auto matchUL = [&](uint32_t st) -> uint32_t {     // UPPER* LOWER+
                    uint32_t i = detail::ascii_upper_run(t, st, L);
                    if (i < L && t[i] >= 0x80) { hitmb = true; return 0; }
                    uint32_t j = detail::ascii_lower_run(t, i, L);
                    if (j < L && t[j] >= 0x80) { hitmb = true; return 0; }
                    return (j > i) ? j : 0;
                };
                auto matchUpL = [&](uint32_t st) -> uint32_t {    // UPPER+ LOWER*
                    uint32_t i = detail::ascii_upper_run(t, st, L);
                    if (i == st) return 0;
                    if (i < L && t[i] >= 0x80) { hitmb = true; return 0; }
                    uint32_t j = detail::ascii_lower_run(t, i, L);
                    if (j < L && t[j] >= 0x80) { hitmb = true; return 0; }
                    return j;
                };
                bool prefelig = !a_let(b0) && !a_dig(b0) && b0 != '\r' && b0 != '\n';
                if (prefelig && p + 1 < L && t[p+1] >= 0x80) { fb = true; break; }
                uint32_t e = 0;
                if (prefelig && p + 1 < L) e = matchUL(p + 1);
                if (hitmb) { fb = true; break; }
                if (!e) e = matchUL(p);
                if (hitmb) { fb = true; break; }
                if (!e && prefelig && p + 1 < L) e = matchUpL(p + 1);
                if (hitmb) { fb = true; break; }
                if (!e) e = matchUpL(p);
                if (hitmb) { fb = true; break; }
                if (e) { e = detail::o_contraction(t, e, L); emit_word(p, e - p); adv = e - p; break; }
                if (a_dig(b0)) {                                   // \p{N}{1,3}
                    uint32_t q = p + 1, c = 1;
                    while (q < L && c < 3 && a_dig(t[q])) { q++; c++; }
                    if (c < 3 && q < L && t[q] >= 0x80) { fb = true; break; }
                    emit(p, q - p); adv = q - p; break;
                }
                {                                                  //  ?punct+[\r\n/]*
                    uint32_t q = p + (b0 == ' ' ? 1 : 0), s4 = q;
                    while (q < L && a_pun(t[q])) q++;
                    if (q > s4) {
                        if (q < L && t[q] >= 0x80) { fb = true; break; }
                        while (q < L && (t[q]=='\r' || t[q]=='\n' || t[q]=='/')) q++;
                        emit(p, q - p); adv = q - p; break;
                    }
                }
                if (a_ws(b0)) {                                    // \s*[\r\n]+|\s+(?!\S)|\s+
                    uint32_t e2 = p, lastnl = UINT32_MAX;
                    while (e2 < L && a_ws(t[e2])) { if (t[e2]=='\r' || t[e2]=='\n') lastnl = e2; e2++; }
                    if (e2 < L && t[e2] >= 0x80) { fb = true; break; }
                    uint32_t plen;
                    if (lastnl != UINT32_MAX) plen = lastnl + 1 - p;
                    else if (e2 == L) plen = e2 - p;
                    else if (e2 - p > 1) plen = e2 - p - 1;
                    else plen = 1;
                    emit(p, plen); adv = plen; break;
                }
                fb = true;                                          // unreachable for ASCII
            } while (0);
            if (fb) { uint32_t l = detail::pretok_next_o200k(U, t, p, L); if (!l) l = 1;
                      merge_piece(V, t + p, l, out); p += l; }
            else p += adv;
        }
        return;
    }
    // cl100k / Llama-3 / Qwen product machine. Shared letter/contraction/punct
    // grammar; per-flag: Qwen single-digit numbers, o200k-style ws for l3/qw.
    const UClass& U = Uc;
    while (p < L) {
        const uint8_t b0 = t[p];
        bool fb = (b0 >= 0x80);
        uint32_t adv = 0;
        if (!fb) do {
            if (b0 == '\'' && p + 1 < L) {                        // '(?i:[sdmt]|ll|ve|re)
                uint8_t c1 = t[p+1] | 0x20;
                if (c1=='s'||c1=='d'||c1=='m'||c1=='t') { emit(p, 2); adv = 2; break; }
                if (p + 2 < L) { uint8_t c2 = t[p+2] | 0x20;
                    if ((c1=='l'&&c2=='l')||(c1=='v'&&c2=='e')||(c1=='r'&&c2=='e')) { emit(p, 3); adv = 3; break; } }
            }
            {                                                      // [^\r\n\p{L}\p{N}]?+\p{L}++
                uint32_t ls;
                if (a_let(b0)) ls = p;
                else if (b0 < 0x80 && !a_dig(b0) && b0 != '\r' && b0 != '\n'
                         && p + 1 < L && a_let(t[p+1])) ls = p + 1;
                else goto not_word;
                {
                    uint32_t we = detail::ascii_letter_run(t, ls, L);
                    if (we < L && t[we] >= 0x80) { fb = true; break; }
                    emit_word(p, we - p); adv = we - p; break;
                }
            }
            not_word:
            if (a_dig(b0)) {                                       // \p{N}{1,3} (Qwen: \p{N})
                uint32_t q = p + 1, c = 1;
                if (!qw) while (q < L && c < 3 && a_dig(t[q])) { q++; c++; }
                if (!qw && c < 3 && q < L && t[q] >= 0x80) { fb = true; break; }
                emit(p, q - p); adv = q - p; break;
            }
            {                                                      //  ?punct+[\r\n]*
                uint32_t q = p + (b0 == ' ' ? 1 : 0), s4 = q;
                while (q < L && a_pun(t[q])) q++;
                if (q > s4) {
                    if (q < L && t[q] >= 0x80) { fb = true; break; }
                    while (q < L && (t[q]=='\r' || t[q]=='\n')) q++;
                    emit(p, q - p); adv = q - p; break;
                }
            }
            if (a_ws(b0)) {                                        // ws cascade, per-family order
                uint32_t e2 = p, lastnl = UINT32_MAX;
                while (e2 < L && a_ws(t[e2])) { if (t[e2]=='\r' || t[e2]=='\n') lastnl = e2; e2++; }
                if (e2 < L && t[e2] >= 0x80) { fb = true; break; }
                uint32_t plen;
                if (l3 || qw) {                                    // o200k-style order
                    if (lastnl != UINT32_MAX) plen = lastnl + 1 - p;
                    else if (e2 == L) plen = e2 - p;
                    else if (e2 - p > 1) plen = e2 - p - 1;
                    else plen = 1;
                } else {                                           // cl100k order
                    if (e2 == L) plen = e2 - p;
                    else if (lastnl != UINT32_MAX) plen = lastnl + 1 - p;
                    else if (e2 - p > 1) plen = e2 - p - 1;
                    else plen = 1;
                }
                emit(p, plen); adv = plen; break;
            }
            fb = true;                                              // unreachable for ASCII
        } while (0);
        if (fb) { uint32_t l = qw ? detail::pretok_next_qwen(U, t, p, L)
                            : l3 ? detail::pretok_next_llama3(U, t, p, L)
                                 : detail::pretok_next(U, t, p, L);
                  merge_piece(V, t + p, l, out); p += l; }
        else p += adv;
    }
}

void Tokenizer::encode(const uint8_t* t, size_t len, std::vector<uint32_t>& out) const {
    if (len > 0xFFFFFFFFull)
        throw std::invalid_argument("quicktok: input exceeds 4 GiB per encode() call — split it");
    size_t start = out.size();
    // NFC-normalizing encodings (Qwen): one cheap scan; only genuinely dirty
    // input (rare) pays for a normalized copy. The normalized buffer goes to the
    // core directly — NFC output can still contain combining marks, so it must
    // not re-enter this check.
    if (impl->nfc && !impl->nfcT.clean(t, len)) {
        std::string norm;
        norm.reserve(len + 16);
        impl->nfcT.normalize(t, len, norm);
        if (norm.size() > 0xFFFFFFFFull)
            throw std::invalid_argument("quicktok: input exceeds 4 GiB after NFC normalization — split it");
        encode_core(impl->V, impl->U, impl->UO, impl->o200k, impl->llama3, impl->qwen, impl->tekken,
                    (const uint8_t*)norm.data(), (uint32_t)norm.size(), out);
    } else {
        encode_core(impl->V, impl->U, impl->UO, impl->o200k, impl->llama3, impl->qwen, impl->tekken,
                    t, (uint32_t)len, out);
    }
    // encodings whose model ids are vocab rank + a constant (Tekken's 1000
    // reserved control slots): translate once on output
    if (impl->id_offset)
        for (size_t i = start; i < out.size(); i++) out[i] += impl->id_offset;
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
                                                           unsigned threads, bool with_special) const {
    std::vector<std::vector<uint32_t>> out(texts.size());
    if (texts.empty()) return out;
    auto enc1 = [&](std::string_view s) {
        return with_special ? encode_with_special(s) : encode(s);
    };
    unsigned hw = std::thread::hardware_concurrency();
    unsigned n = threads ? threads : (hw ? hw : 4);
    if (n > texts.size()) n = (unsigned)texts.size();
    if (n <= 1) {
        for (size_t i = 0; i < texts.size(); i++) out[i] = enc1(texts[i]);
        return out;
    }
    std::atomic<size_t> next{0};
    std::vector<std::thread> ths;
    ths.reserve(n);
    for (unsigned t = 0; t < n; t++)
        ths.emplace_back([&] {
            for (size_t i; (i = next.fetch_add(1, std::memory_order_relaxed)) < texts.size(); )
                out[i] = enc1(texts[i]);
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
    const uint32_t off = impl->id_offset;
    for (size_t i = 0; i < n; i++) {
        uint32_t id = ids[i] - off;          // off=0 for builtin encodings
        if (ids[i] >= off && id < V.n) {
            out.append((const char*)V.all.data() + V.tstart[id], V.tstart[id+1] - V.tstart[id]);
            continue;
        }
        for (const auto& [s, sid] : impl->specials)
            if (sid == ids[i]) { out.append(s); break; }   // specials carry MODEL ids (unshifted)
        // other out-of-range ids are skipped
    }
}
std::string Tokenizer::decode(const std::vector<uint32_t>& ids) const {
    std::string out; out.reserve(ids.size() * 4);
    decode(ids.data(), ids.size(), out);
    return out;
}

}  // namespace quicktok
