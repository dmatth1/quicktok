// Faithful C++ port of the `bpe` crate's encode_via_backtracking (the SOTA, 45
// MB/s merge-only). Defines `struct Vocab` for VOCAB=6. Only needs:
//   next_match       longest vocab token that is a prefix of text  (trie)
//   next_prefix[id]  longest proper-prefix token of token id
//   split_table[id]  the two tokens id was merged from (or (id,id) if original)
//   pair_lookup      (a,b) -> merged token, if it exists
//   is_valid_token_pair  merge-reversal compatibility check
// Backtracking driver + BitField(all-1s) reproduce bpe exactly. Trie edges use a
// flat open-addressing table (fast O(1)-ish step), not Aho-Corasick.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>

static const uint32_t RANK_MAX = 0xFFFFFFFFu;

// --- portability shims (the kernel targets GCC/Clang; MSVC for Windows wheels) ---
#if defined(__GNUC__) || defined(__clang__)
#  define QT_UNLIKELY(x) __builtin_expect(!!(x), 0)
namespace quicktok { namespace detail {
template <class T> static inline T qt_relaxed_load(const T& x) { return __atomic_load_n(&x, __ATOMIC_RELAXED); }
template <class T> static inline void qt_relaxed_store(T& x, T v) { __atomic_store_n(&x, v, __ATOMIC_RELAXED); }
}}
#else
#  define QT_UNLIKELY(x) (x)
namespace quicktok { namespace detail {
// MSVC / x86 / ARM: an aligned load/store of a <=8-byte value is not torn in
// practice; and the memo is a pure-function cache, so even a torn value only
// fails the tag check and triggers a recompute — the result is always correct.
template <class T> static inline T qt_relaxed_load(const T& x) { return x; }
template <class T> static inline void qt_relaxed_store(T& x, T v) { x = v; }
}}
#endif

#ifndef IVBITS
#define IVBITS 20   // memo capacity (2^20 peak on M1, re-confirmed under TRIE2; x86 uses 21)
#endif
#define IVTAGBITS (34 - IVBITS)
static_assert(IVTAGBITS >= 0 && IVTAGBITS <= 14, "IVDENSE u16 slot: need 34-IVBITS <= 14 tag bits");
#ifndef IVBITS_W
// wide-id (36-bit key) memo capacity; u32 slots. Misses are expensive (ivtp_slow
// chains), so capacity beats footprint until the memo crowds the LLC: x86 sweeps
// best at 2^22 (16 MB, vs 33 MB L3 on the bench Xeon; 2^23 regresses). Non-x86
// keeps 2^20 (4 MB) — same capacity as the u64 memo it replaced, half the bytes.
#if defined(__x86_64__) || defined(_M_X64)
#define IVBITS_W 22
#else
#define IVBITS_W 20
#endif
#endif
static_assert(36 - IVBITS_W >= 0 && 36 - IVBITS_W <= 30, "IVWIDE u32 slot: need 36-IVBITS_W <= 30 tag bits");

namespace quicktok {
namespace detail {

// odd-token side-table layout: 18-bit token field (fits both cl100k's 100,256 and
// o200k's 199,998 ids). slot = (key+1)<<19 | hasdeeper<<18 | token.
constexpr uint32_t OTAB_TOKMASK = (1u << 18) - 1;
constexpr uint64_t OTAB_DEEPBIT = 1ull << 18;
constexpr int      OTAB_KEYSH   = 19;
constexpr uint32_t OTAB_LOWMASK = (1u << 19) - 1;   // hasdeeper|token (odd_lookup return)

struct Vocab {
    std::vector<uint8_t> all;            // token bytes concatenated, by id
    std::vector<uint32_t> tstart;        // tstart[id]..tstart[id+1]
    std::vector<uint8_t> tlen;           // token byte-length (max 128) — L1-hot, 1 load vs 2 in tstart
    uint32_t n = 0;
    std::unordered_map<std::string, uint32_t> b2id;
    // trie edges: open-addressing (key = node*256+byte -> child node)
    std::vector<uint64_t> etab; uint32_t emask = 0;   // slot: (key+1)<<32 | child; 0=empty. key=node<<8|byte
    uint32_t root_child[256];            // direct edges from root (hottest: next_match restarts here)
    std::vector<uint32_t> r2node, r2best; // 65536: node after 2 bytes / deepest token in first <=2 bytes
    std::vector<uint32_t> tnode_tok;     // per trie node: token id ending here, or MAX
    std::vector<uint32_t> npm;           // next_prefix_match
    std::vector<std::pair<uint32_t,uint32_t>> split;
    // flat open-addressing (t1<<32|t2)+1 -> merged token; 0 = empty
    std::vector<uint64_t> plk; std::vector<uint32_t> plv; uint32_t plmask = 0;
    // tagged direct-mapped memo for is_valid_token_pair (adjacent pairs recur in text).
    // One u64/slot packs key+result: slot = ((mkey+1)<<1)|result, 0=empty. mkey only
    // uses 34 bits (t1,t2 < 2^17) so the <<1 never overflows. ONE load/hit (was two:
    // key u64 + result u8) — M1-neutral (L1-hot) but an x86 win (memo lives in L2).
    // dense memo: u16/slot, same slot COUNT as the u64 memo -> 4x smaller (2MB at
    // 2^20), 4x more entries per cache line. mkey34=(t1<<17|t2) goes through a
    // BIJECTIVE 34-bit mixer (self-inverse xorshift / odd multiply / xorshift), so
    // index(IVBITS)+tag(34-IVBITS) reconstruct the pair exactly — no aliasing, the
    // memo stays exact. slot = 0x8000 | tag<<1 | result; 0 = empty.
    mutable std::vector<uint16_t> ivm; uint32_t ivmask = 0;
    // wide-id memo (vocabs with ids >= 2^17, e.g. o200k): same dense bijective-
    // mixer scheme widened to a 36-bit pair key (ids < 2^18, enforced at load).
    // u32 slots = 0x80000000 | tag<<1 | result (tag = 36-IVBITS_W bits); 4 MB at
    // 2^20 — half the u64 memo it replaces, 2x entries per cache line. Selected
    // once per Vocab at load; the hot dense path is untouched and the wide path
    // lives in trie2_mb.cpp (own TU — see the inlining note above encode_mb).
    mutable std::vector<uint32_t> ivmw;
    bool wide_ids = false;
    bool ivtp_wide(uint32_t t1, uint32_t t2) const;   // trie2_mb.cpp
    // 2-byte-radix trie: walk consumes 2 bytes per ONE aligned-16B slot load (key +
    // child + step-best-token packed). Odd-depth tokens via the side table otab,
    // probed only on 2-byte miss / trailing byte. r3 resolves 3-byte UTF-8 starts
    // (lead E0..EF) with zero probes. The byte trie (etab/tnode_tok) is used at
    // construction and for len==1; cold during encode.
    struct alignas(16) E2 { uint64_t key; uint64_t val; };   // key=(node<<16|b1b2)+1, 0=empty; val=child<<32|best32
    std::vector<E2> e2; uint32_t e2mask = 0;
    std::vector<uint64_t> otab; uint32_t omask = 0;          // slot=((node<<8|b)+1)<<18 | hasdeeper<<17 | tok17
    std::vector<uint32_t> r3node, r3best;
    // whole-piece encoder for pieces starting with a 3-byte UTF-8 lead (trie2_mb.cpp)
    void encode_mb(const uint8_t* text, uint32_t len, std::vector<uint32_t>& out) const;

    size_t size() const { return n; }
    inline uint32_t token_len(uint32_t id) const { return tlen[id]; }
    inline uint32_t pl_get(uint64_t key) const {
        uint32_t i = (uint32_t)(key * 0x9E3779B97F4A7C15ull >> 40) & plmask;
        while (plk[i]) { if (plk[i] == key + 1) return plv[i]; i = (i+1) & plmask; }
        return RANK_MAX;
    }
    inline void pl_put(uint64_t key, uint32_t val) {
        uint32_t i = (uint32_t)(key * 0x9E3779B97F4A7C15ull >> 40) & plmask;
        while (plk[i]) { if (plk[i] == key + 1) { plv[i] = val; return; } i = (i+1) & plmask; }
        plk[i] = key + 1; plv[i] = val;
    }

    // ---- edge-slot pack/unpack ----
    // [63..32]=key+1, [31..0]=child.
    // [36..18]=child(19b) [17..0]=token(18b, NONE=0x3FFFF) — folds tnode_tok[child] into the
    // slot so the trie walk does ONE load/step (the dependent tnode_tok load is gone). M1's
    // v11 rejected this (its tnode_tok load was L2-cheap); on x86 that load is L3, so it pays.
    static constexpr uint32_t ETOKNONE = 0x3FFFF;
    static inline uint64_t emk(uint32_t kp1, uint32_t child, uint32_t){ return ((uint64_t)kp1<<32)|child; }
    static inline uint32_t es_kp1(uint64_t s){ return (uint32_t)(s>>32); }
    static inline uint32_t es_child(uint64_t s){ return (uint32_t)s; }

    // ---- trie ----
    inline uint32_t edge(uint32_t node, uint8_t b) const {
        if (node == 0) return root_child[b];
        uint64_t k = ((uint64_t)node << 8) | b; uint32_t i = (uint32_t)(k * 0x9E3779B97F4A7C15ull >> 40) & emask;
        uint32_t key = (uint32_t)(k + 1);
        while (etab[i]) { if (es_kp1(etab[i]) == key) return es_child(etab[i]); i = (i+1) & emask; }
        return 0;  // node 0 = root = "no edge" (root is never a child)
    }
    inline uint32_t odd_lookup(uint32_t node, uint8_t b) const {  // (hasdeeper|tok) or MAX if absent
        uint64_t k = ((uint64_t)node << 8) | b;
        uint32_t i = (uint32_t)(k * 0x9E3779B97F4A7C15ull >> 40) & omask;
        while (otab[i]) { if ((otab[i] >> OTAB_KEYSH) == (k + 1)) return (uint32_t)(otab[i] & OTAB_LOWMASK); i = (i+1) & omask; }
        return RANK_MAX;
    }
    // longest token that is a prefix of text[0..len) — 2-byte steps, 1 slot load each.
    // One predictable branch at entry routes 3-byte UTF-8 starts to the multibyte
    // walk; the ASCII hot loop below is the original TRIE2 walk, untouched.
    // NOTE: deliberately NO multibyte dispatch in here. Any extra branch/call in
    // this function (or even an unreachable sibling walk in this TU) flips clang's
    // inlining of the hot path and costs ~14% on Latin (measured 2026-06-09).
    // Multibyte pieces are routed per-piece in merge_piece -> encode_mb (own TU).
    inline uint32_t next_match(const uint8_t* text, uint32_t len) const {
        if (len == 0) return RANK_MAX;
        if (len == 1) { uint32_t n = root_child[text[0]]; return n ? tnode_tok[n] : RANK_MAX; }
        uint32_t idx = ((uint32_t)text[0] << 8) | text[1];
        uint32_t node = r2node[idx], best = r2best[idx];
        uint32_t i = 2;
        while (node && i + 1 < len) {
            uint64_t k = ((uint64_t)node << 16) | ((uint32_t)text[i] << 8) | text[i+1];
            uint32_t h = (uint32_t)(k * 0x9E3779B97F4A7C15ull >> 40) & e2mask;
            uint64_t key = k + 1, val = 0;
            while (e2[h].key) { if (e2[h].key == key) { val = e2[h].val; break; } h = (h+1) & e2mask; }
            if (!val) {  // no 2-byte step: an odd-depth token may still extend one byte
                uint32_t o = odd_lookup(node, text[i]); if (o != RANK_MAX) best = o & OTAB_TOKMASK;
                return best;
            }
            uint32_t b32 = (uint32_t)val; if (b32 != RANK_MAX) best = b32;
            node = (uint32_t)(val >> 32); i += 2;
        }
        if (node && i < len) { uint32_t o = odd_lookup(node, text[i]); if (o != RANK_MAX) best = o & OTAB_TOKMASK; }
        return best;
    }
    inline uint32_t next_prefix(uint32_t id) const { return npm[id]; }

    bool is_valid_token_pair(uint32_t t1, uint32_t t2) const {
        if (QT_UNLIKELY(wide_ids)) return ivtp_wide(t1, t2);
        uint64_t mk = ((uint64_t)t1 << 17) | t2;                    // 34-bit pair key
        uint64_t m = mk ^ (mk >> 17);
        m = (m * ((0x9E3779B97F4A7C15ull & 0x3FFFFFFFFull) | 1)) & 0x3FFFFFFFFull;
        m ^= m >> 17;                                               // bijection done
        uint32_t h = (uint32_t)(m >> IVTAGBITS);
        uint16_t want = (uint16_t)(0x8000u | (((uint32_t)m & ((1u<<IVTAGBITS)-1)) << 1));
        // relaxed atomics: each slot value is self-consistent (tag+result written in
        // one u16 store), so concurrent encode() on the same Tokenizer is safe — a
        // racing thread sees either the old or the new entry, both correct.
        uint16_t s = qt_relaxed_load(ivm[h]);                       // one load
        if ((uint16_t)(s & 0xFFFEu) == want) return s & 1;
        bool res = ivtp_slow(t1, t2);
        qt_relaxed_store(ivm[h], (uint16_t)(want | (uint16_t)res));
        return res;
    }
    bool ivtp_slow(uint32_t t1, uint32_t t2) const {
        uint32_t limit = RANK_MAX;
        for (;;) {
            uint32_t c = pl_get(((uint64_t)t1 << 32) | t2);
            if (c != RANK_MAX && c < limit) return false;
            if (t1 > t2) {
                limit = t1; t1 = split[t1].second;
                if (t1 == limit) { limit = t2 + 1; t2 = split[t2].first; if (t2 + 1 == limit) return true; }
            } else {
                limit = t2 + 1; t2 = split[t2].first;
                if (t2 + 1 == limit) { limit = t1; t1 = split[t1].second; if (t1 == limit) return true; }
            }
        }
    }

    // ---- backtracking encode (faithful port of BacktrackEncoder) ----
    void encode(const uint8_t* text, uint32_t len, std::vector<uint32_t>& out) const {
        if (len == 0) return;
        encode_with_first(text, len, next_match(text, len), out);
    }
    // encode using a precomputed first = next_match(text,len) (lets the e2e fused
    // loop reuse the walk it already did instead of recomputing it).
    void encode_with_first(const uint8_t* text, uint32_t len, uint32_t first, std::vector<uint32_t>& out) const {
        if (len == 0) return;
        size_t out_start = out.size();
        // GREEDY fast path: backtracking only ever backtracks when is_valid fails (rare),
        // so for any piece that tokenizes greedily, the bitfield+machinery is pure overhead.
        { uint32_t pos = 0, last = RANK_MAX, nt = first; bool ok = true;
          while (pos < len) {
              uint32_t token = nt;
              if (last != RANK_MAX && !is_valid_token_pair(last, token)) { ok = false; break; }
              out.push_back(token); last = token; pos += token_len(token);
              nt = (pos < len) ? next_match(text + pos, len - pos) : RANK_MAX;
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
                    next_token = (pos < len) ? next_match(text + pos, len - pos) : RANK_MAX;
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

    // ---- construction ----
    inline uint32_t edge_build(uint32_t node, uint8_t b) {  // get-or-create child
        if (node == 0) {
            if (root_child[b]) return root_child[b];
            uint32_t c = (uint32_t)tnode_tok.size(); tnode_tok.push_back(RANK_MAX); root_child[b] = c; return c;
        }
        uint64_t k = ((uint64_t)node << 8) | b; uint32_t i = (uint32_t)(k * 0x9E3779B97F4A7C15ull >> 40) & emask;
        uint32_t key = (uint32_t)(k+1);
        while (etab[i]) { if (es_kp1(etab[i]) == key) return es_child(etab[i]); i = (i+1) & emask; }
        uint32_t child = (uint32_t)tnode_tok.size(); tnode_tok.push_back(RANK_MAX);
        etab[i] = emk(key, child, ETOKNONE); return child;
    }
    inline uint32_t find_id(const uint8_t* p, uint32_t l) const {
        auto it = b2id.find(std::string((const char*)p, l)); return it==b2id.end()? RANK_MAX : it->second;
    }

    static Vocab load(const char* path) {
        FILE* f = fopen(path, "rb");
        if (!f) throw std::runtime_error(std::string("quicktok: cannot open vocab file: ") + path);
        auto fail = [&](const char* why) {
            fclose(f);
            throw std::runtime_error(std::string("quicktok: bad vocab file (") + why + "): " + path);
        };
        uint32_t n; if (fread(&n,4,1,f)!=1) fail("truncated header");
        // token ids must fit the 17-bit packing used by the odd-token table and the
        // 34-bit memo pair key (cl100k = 100,256 tokens; o200k needs a wider build).
        if (n == 0 || n > (1u<<18)) fail("token count out of range (max 262144 ids)");
        Vocab V; V.n = n;
        std::vector<std::string> tb(n);
        std::vector<bool> seen(n, false);
        std::string buf;
        for (uint32_t k=0;k<n;k++){ uint16_t bl; if(fread(&bl,2,1,f)!=1) fail("truncated record");
            if (bl == 0 || bl > 255) fail("token length out of range");
            buf.resize(bl); if(fread(buf.data(),1,bl,f)!=bl) fail("truncated token bytes");
            uint32_t r; if(fread(&r,4,1,f)!=1) fail("truncated rank");
            if (r >= n) fail("rank out of range");
            if (seen[r]) fail("duplicate rank");
            seen[r] = true; tb[r]=buf; }
        { uint8_t extra; if (fread(&extra,1,1,f)==1) fail("trailing bytes"); }
        fclose(f);
        // all/tstart + b2id, by id order
        V.tstart.push_back(0);
        for (uint32_t id=0; id<n; id++){ V.all.insert(V.all.end(), tb[id].begin(), tb[id].end());
            V.tstart.push_back((uint32_t)V.all.size()); V.b2id.emplace(tb[id], id); }
        V.tlen.resize(n); for (uint32_t id=0; id<n; id++) V.tlen[id] = (uint8_t)(V.tstart[id+1]-V.tstart[id]);
        // trie
        uint32_t ecap = 1; while (ecap < V.all.size()*2) ecap <<= 1; if (ecap<1024) ecap=1024;
        V.etab.assign(ecap,0); V.emask = ecap-1;
        memset(V.root_child, 0, sizeof(V.root_child));
        V.tnode_tok.assign(1, RANK_MAX);  // root = node 0
        for (uint32_t id=0; id<n; id++){ uint32_t node=0; const std::string& t=tb[id];
            for (size_t j=0;j<t.size();j++) node = V.edge_build(node, (uint8_t)t[j]);
            V.tnode_tok[node] = id; }
        // rehash the (oversized, sparse) edge table to a cache-tight size. Target load
        // factor is tunable per-arch: M1 default 0.45 (~4.2MB, L2-resident, ~1 probe);
        // x86 has a smaller per-core L2 so the size/probe tradeoff differs — sweep via
        // -DEDGE_LOAD=0.x (see X86_PORT_LOG.md).
#ifndef EDGE_LOAD
#define EDGE_LOAD 0.45
#endif
        { uint32_t want = 1024; while ((double)V.tnode_tok.size()/want > (EDGE_LOAD)) want <<= 1;
          if (want < V.emask+1) {
            std::vector<uint64_t> ne(want,0); uint32_t nm = want-1;
            for (uint32_t i=0;i<=V.emask;i++) if (V.etab[i]) {
                uint64_t k = (uint64_t)es_kp1(V.etab[i])-1; uint32_t j=(uint32_t)(k*0x9E3779B97F4A7C15ull>>40)&nm;
                while (ne[j]) j=(j+1)&nm; ne[j]=V.etab[i]; }
            V.etab.swap(ne); V.emask = nm; } }
        // 2-level direct table: node + deepest-token after the first <=2 bytes
        V.r2node.assign(65536, 0); V.r2best.assign(65536, RANK_MAX);
        for (uint32_t b0=0; b0<256; b0++){ uint32_t n1 = V.root_child[b0]; if (!n1) continue;
            uint32_t tok1 = V.tnode_tok[n1];
            for (uint32_t b1=0; b1<256; b1++){ uint32_t idx=(b0<<8)|b1; uint32_t n2 = V.edge(n1,(uint8_t)b1);
                if (n2){ V.r2node[idx]=n2; uint32_t t2=V.tnode_tok[n2]; V.r2best[idx]=(t2!=RANK_MAX)?t2:tok1; }
                else { V.r2node[idx]=0; V.r2best[idx]=tok1; } } }
        // 2-byte trie + odd-token side table, derived from the (complete) byte trie.
        // e2 target load factor, tunable per-arch like EDGE_LOAD (-DE2_LOAD=0.x):
        // pow2 sizing means the real choices are ~0.39 (default) vs ~0.78 (half the
        // footprint, ~2x probes) — the footprint matters once the vocab outgrows L2.
#ifndef E2_LOAD
#define E2_LOAD 0.45
#endif
        { size_t ne2 = 0, nodd = 0;
          // first pass: count
          for (uint32_t id=0; id<n; id++){ uint32_t L=V.tlen[id]; if (L>=4) ne2 += (L-2)/2; if (L>=3 && (L&1)) nodd++; }
          uint32_t cap2 = 1024; while ((double)ne2/cap2 > (E2_LOAD)) cap2 <<= 1;     // upper bound (dups counted)
          uint32_t capo = 1024; while ((double)nodd/capo > 0.45) capo <<= 1;
          V.e2.assign(cap2, E2{0,0}); V.e2mask = cap2-1;
          V.otab.assign(capo, 0);     V.omask  = capo-1;
          std::vector<uint32_t> path;
          for (uint32_t id=0; id<n; id++){
              const uint8_t* t = V.all.data() + V.tstart[id];
              uint32_t L = V.tlen[id];
              path.assign(L+1, 0); uint32_t node = 0;
              for (uint32_t j=0;j<L;j++){ node = (node==0)? V.root_child[t[j]] : V.edge(node, t[j]); path[j+1]=node; }
              for (uint32_t d=2; d+2<=L; d+=2){
                  uint64_t k = ((uint64_t)path[d] << 16) | ((uint32_t)t[d] << 8) | t[d+1];
                  uint32_t h = (uint32_t)(k * 0x9E3779B97F4A7C15ull >> 40) & V.e2mask;
                  while (V.e2[h].key) { if (V.e2[h].key == k+1) break; h = (h+1) & V.e2mask; }
                  if (!V.e2[h].key) {
                      uint32_t bt2 = V.tnode_tok[path[d+2]], bt1 = V.tnode_tok[path[d+1]];
                      uint32_t best = (bt2 != RANK_MAX) ? bt2 : bt1;
                      V.e2[h] = E2{k+1, ((uint64_t)path[d+2] << 32) | best};
                  }
              }
              if (L>=3 && (L&1)) {   // token ends at odd depth: side-table entry from its even parent
                  uint64_t k = ((uint64_t)path[L-1] << 8) | t[L-1];
                  uint32_t h = (uint32_t)(k * 0x9E3779B97F4A7C15ull >> 40) & V.omask;
                  while (V.otab[h]) { if ((V.otab[h] >> OTAB_KEYSH) == k+1) break; h = (h+1) & V.omask; }
                  if (!V.otab[h]) V.otab[h] = ((k+1) << OTAB_KEYSH) | id;   // hasdeeper flag OR'd in later
              }
          }
          // ne2 over-counted duplicate edges (shared prefixes) -> rehash down to the
          // tightest power-of-2 at <=0.45 load over the DISTINCT count (byte-trie pattern).
          { size_t used2=0; for (auto& s : V.e2) if (s.key) used2++;
            uint32_t want = 1024; while ((double)used2/want > (E2_LOAD)) want <<= 1;
            if (want < V.e2mask+1) {
                std::vector<E2> ne(want, E2{0,0}); uint32_t nm = want-1;
                for (uint32_t i=0;i<=V.e2mask;i++) if (V.e2[i].key) {
                    uint64_t k = V.e2[i].key - 1; uint32_t j = (uint32_t)(k*0x9E3779B97F4A7C15ull>>40)&nm;
                    while (ne[j].key) j = (j+1)&nm; ne[j] = V.e2[i]; }
                V.e2.swap(ne); V.e2mask = nm; } }
        }
        // next_prefix_match[id] = longest proper-prefix token. Direct byte-trie walk
        // (== next_match(t, len-1)); avoids depending on TRIE2 tables built later.
        V.npm.resize(n);
        for (uint32_t id=0; id<n; id++){ const std::string& t=tb[id];
            uint32_t node=0, best=RANK_MAX;
            for (size_t j=0; j+1<t.size(); j++){
                node = (node==0 && j==0) ? V.root_child[(uint8_t)t[0]] : V.edge(node,(uint8_t)t[j]);
                if (!node) break;
                uint32_t tk = V.tnode_tok[node]; if (tk != RANK_MAX) best = tk;
            }
            V.npm[id] = best; }
        // split_table + pair_lookup (ids in rank order)
        uint32_t pcap = 1; while (pcap < n*2) pcap <<= 1; V.plk.assign(pcap,0); V.plv.assign(pcap,0); V.plmask = pcap-1;

        uint32_t icap = 1u<<IVBITS; V.ivmask = icap-1;
        V.wide_ids = (n > (1u<<17));            // ids past 17 bits: the 34-bit mixer can't pack the pair key
        if (V.wide_ids) V.ivmw.assign(1u<<IVBITS_W,0); else V.ivm.assign(icap,0);
        V.split.reserve(n);
        for (uint32_t id=0; id<n; id++){ const std::string& t=tb[id];
            uint32_t token1 = V.npm[id]; bool done=false;
            while (token1 != RANK_MAX) {
                uint32_t l1 = V.token_len(token1);
                uint32_t token2 = V.find_id((const uint8_t*)t.data()+l1, (uint32_t)t.size()-l1);
                if (token2 != RANK_MAX && token1 < id && token2 < id
                    && V.is_valid_token_pair(token1, token2)) {
                    V.pl_put(((uint64_t)token1<<32)|token2, id);
                    V.split.push_back({token1, token2}); done=true; break;
                }
                token1 = V.npm[token1];
            }
            if (!done) V.split.push_back({id, id});
        }
        // is_valid_token_pair is only PURE once split_table/pair_lookup are complete;
        // construction-time calls populated the memo with stale (incomplete-table) results. Clear it.
        std::fill(V.ivm.begin(), V.ivm.end(), 0);
        std::fill(V.ivmw.begin(), V.ivmw.end(), 0);
        // Built LAST on purpose: allocating these mid-load shifted the heap layout of
        // the hot tables (e2/otab/plk/ivm) and cost ~13% on Latin (L2 physical-conflict
        // luck, measured 2026-06-09). Keep the v1 allocation sequence; append new tables.
        { std::vector<uint8_t> haschild(V.tnode_tok.size(), 0);
          for (uint32_t i=0;i<=V.emask;i++) if (V.etab[i])
              haschild[(uint32_t)(((uint64_t)es_kp1(V.etab[i])-1) >> 8)] = 1;
          for (uint32_t h=0; h<=V.omask; h++) if (V.otab[h]) {           // OR hasdeeper into otab
              uint64_t k = (V.otab[h] >> OTAB_KEYSH) - 1;
              uint32_t m = V.edge((uint32_t)(k >> 8), (uint8_t)(k & 255));
              if (m && haschild[m]) V.otab[h] |= OTAB_DEEPBIT;
          }
          V.r3node.assign(1u<<16, 0); V.r3best.assign(1u<<16, RANK_MAX);  // r3 direct table
          for (uint32_t b0 = 0xE0; b0 <= 0xEF; b0++)
            for (uint32_t b1 = 0x80; b1 <= 0xBF; b1++) {
              uint32_t idx2 = (b0<<8)|b1; uint32_t n2 = V.r2node[idx2]; uint32_t bs2 = V.r2best[idx2];
              for (uint32_t b2 = 0x80; b2 <= 0xBF; b2++) {
                  uint32_t i3 = ((b0&0xF)<<12)|((b1&0x3F)<<6)|(b2&0x3F);
                  uint32_t best = bs2;
                  if (n2) { uint32_t m = V.edge(n2, (uint8_t)b2);
                            if (m) { uint32_t tk = V.tnode_tok[m]; if (tk != RANK_MAX) best = tk; } }
                  V.r3node[i3] = n2; V.r3best[i3] = best;
              }
            }
        }
        return V;
    }
};

}  // namespace detail
}  // namespace quicktok
