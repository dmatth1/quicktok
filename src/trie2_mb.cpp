// TRIE2 multibyte-piece encoder — its own TU on purpose. Defining (or even
// declaring+dispatching to) this walk inside the main TU flips clang's inlining
// of the hot ASCII path and costs ~14% on Latin (measured 2026-06-09). merge_piece routes pieces whose first byte is a 3-byte
// UTF-8 lead (E0..EF) here; everything else takes the untouched v1 path.
//
// Compile with the SAME -D flags as the main TU (build.sh does this).
#include <cstdint>
#include <vector>
#include "bpe.hpp"

namespace quicktok {
namespace detail {

// next_match with the multibyte-optimized walk (sim v4, oracle-verified):
//  - 3-byte UTF-8 start: r3 resolves the whole first char with 0 probes
//  - in the loop, bytes >=0x80 probe the odd table FIRST (CJK tokens end at odd
//    depths 3,9,..); the hasdeeper flag lets the walk stop after 1 probe
//  - ASCII bytes inside a mixed piece fall back to the e2-first order
static uint32_t nm_mb(const Vocab& V, const uint8_t* text, uint32_t len) {
    if (len == 0) return RANK_MAX;
    if (len == 1) { uint32_t n = V.root_child[text[0]]; return n ? V.tnode_tok[n] : RANK_MAX; }
    uint32_t node, best, i = 2; bool odd_covered = false;
    // r3 indexes by the codepoint of a 3-byte UTF-8 char (continuation bits masked),
    // so it is valid ONLY for well-formed sequences. For ill-formed bytes (e.g. a
    // truncated stream) the masked index would alias a *different* token, making
    // encode lossy — so require valid continuations and otherwise take the
    // byte-accurate r2 path. (The dominant case, real CJK text, is well-formed.)
    if (text[0] >= 0xE0 && text[0] < 0xF0 && len >= 3
        && (text[1] & 0xC0) == 0x80 && (text[2] & 0xC0) == 0x80) {
        uint32_t i3 = (((uint32_t)text[0] & 0xF) << 12) | (((uint32_t)text[1] & 0x3F) << 6) | ((uint32_t)text[2] & 0x3F);
        node = V.r3node[i3]; best = V.r3best[i3]; odd_covered = true;   // depth<=3 known, 0 probes
    } else {
        uint32_t idx = ((uint32_t)text[0] << 8) | text[1];
        node = V.r2node[idx]; best = V.r2best[idx];
    }
    while (node && i + 1 < len) {
        bool odd_done = odd_covered; odd_covered = false;
        if (!odd_done && text[i] >= 0x80) {            // odd-first: token likely ends at odd depth
            odd_done = true;
            uint32_t o = V.odd_lookup(node, text[i]);
            if (o != RANK_MAX) {
                best = o & OTAB_TOKMASK;
                if (!(o & OTAB_DEEPBIT)) return best;  // no deeper token exists: done in 1 probe
            }
        }
        uint64_t k = ((uint64_t)node << 16) | ((uint32_t)text[i] << 8) | text[i+1];
        uint32_t h = (uint32_t)(k * 0x9E3779B97F4A7C15ull >> 40) & V.e2mask;
        uint64_t key = k + 1, val = 0;
        while (V.e2[h].key) { if (V.e2[h].key == key) { val = V.e2[h].val; break; } h = (h+1) & V.e2mask; }
        if (!val) {
            if (!odd_done) {
                uint32_t o = V.odd_lookup(node, text[i]);
                if (o != RANK_MAX) best = o & OTAB_TOKMASK;
            }
            return best;
        }
        uint32_t b32 = (uint32_t)val; if (b32 != RANK_MAX) best = b32;
        node = (uint32_t)(val >> 32); i += 2;
    }
    if (node && i < len && !odd_covered) {
        uint32_t o = V.odd_lookup(node, text[i]);
        if (o != RANK_MAX) best = o & OTAB_TOKMASK;
    }
    return best;
}

// wide-id (o200k-class) validity memo: classic u64 slot = ((mkey+1)<<1)|result.
// Kept out of the hot TU so the dense cl100k path's codegen is untouched.
bool Vocab::ivtp_wide(uint32_t t1, uint32_t t2) const {
    uint64_t mkey = ((uint64_t)t1 << 32) | t2;
    uint32_t h = (uint32_t)((mkey * 0x9E3779B97F4A7C15ull) >> 32) & ivmask;
    uint64_t want = (mkey + 1) << 1;
    uint64_t s = qt_relaxed_load(ivm64[h]);
    if ((s >> 1) == (mkey + 1)) return s & 1;
    bool res = ivtp_slow(t1, t2);
    qt_relaxed_store<uint64_t>(ivm64[h], want | (uint64_t)res);
    return res;
}

// mirror of Vocab::encode_with_first (greedy fast path + backtracking fallback),
// with every next_match replaced by nm_mb. Exact-correct by the same argument:
// nm_mb == next_match on every input (sim-verified), driver logic identical.
void Vocab::encode_mb(const uint8_t* text, uint32_t len, std::vector<uint32_t>& out) const {
    if (len == 0) return;
    uint32_t first = nm_mb(*this, text, len);
    size_t out_start = out.size();
    { uint32_t pos = 0, last = RANK_MAX, nt = first; bool ok = true;
      while (pos < len) {
          uint32_t token = nt;
          if (last != RANK_MAX && !is_valid_token_pair(last, token)) { ok = false; break; }
          out.push_back(token); last = token; pos += token_len(token);
          nt = (pos < len) ? nm_mb(*this, text + pos, len - pos) : RANK_MAX;
      }
      if (ok) return;
      out.resize(out_start);   // greedy hit an invalid pair -> full backtracking
    }
    static thread_local std::vector<uint32_t> toks;
    static thread_local std::vector<uint64_t> bf;
    toks.clear();
    uint32_t words = (len + 1 + 63) >> 6; bf.assign(words, ~0ull);
    auto is_set = [&](uint32_t b){ return (bf[b>>6] >> (b&63)) & 1ull; };
    auto clr    = [&](uint32_t b){ bf[b>>6] &= ~(1ull << (b&63)); };

    uint32_t pos = 0;
    uint32_t next_token = first;
    while (next_token != RANK_MAX) {
        uint32_t token = next_token;
        uint32_t last = toks.empty() ? RANK_MAX : toks.back();
        for (;;) {
            uint32_t end = pos + token_len(token);
            if (is_set(end) && (last == RANK_MAX || is_valid_token_pair(last, token))) {
                toks.push_back(token); pos = end;
                next_token = (pos < len) ? nm_mb(*this, text + pos, len - pos) : RANK_MAX;
                break;
            }
            uint32_t shorter = next_prefix(token);
            if (shorter != RANK_MAX) { token = shorter; continue; }
            clr(pos);
            if (!toks.empty()) { toks.pop_back(); pos -= token_len(last); }
            next_token = last;
            break;
        }
    }
    out.insert(out.end(), toks.begin(), toks.end());
}

}  // namespace detail
}  // namespace quicktok
