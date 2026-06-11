// Hand-coded pretokenizer for the FIXED cl100k_base regex (the "specialize the
// known pattern, skip the general regex engine" win). Reproduces the 8 alternatives:
//   '(?i:[sdmt]|ll|ve|re) | [^\r\n\p{L}\p{N}]?+\p{L}++ | \p{N}{1,3}+
//   | ?[^\s\p{L}\p{N}]++[\r\n]*+ | \s++$ | \s*[\r\n] | \s+(?!\S) | \s
// Unicode \p{L}/\p{N}/\s from fixtures/uniclass.bin (exact vs the reference engine).
// pretok(text,len, cb): calls cb(offset,length) for each pretoken, in order.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <immintrin.h>
#endif

namespace quicktok {
namespace detail {

// length-advance over a run of ASCII letters [A-Za-z] starting at q, SIMD 16/32B
// per step. Stops at the first non-ASCII-letter byte (incl any byte >=128, which
// the caller handles via u8dec). All ISA paths compute the IDENTICAL predicate as
// the scalar tail — `(uint8_t)((b|0x20)-'a') <= 25` — so they are bit-exact w.r.t.
// each other (cross-ISA verified in pretok_simd_test.cpp). NO_SIMD_LET forces scalar.
//
// x86 (AVX2/SSE2) port of the NEON original: fold to lowercase (| 0x20), d = lo-'a'
// (modular), then saturating-unsigned-subtract 25 — zero iff d<=25 i.e. a letter
// (bytes>=128 fold to large d, never zero). movemask gives a letter-bitmask; the
// first cleared bit (ctz of its complement) is the first non-letter.
static inline uint32_t ascii_letter_run(const uint8_t* t, uint32_t q, uint32_t len) {
// x86 default is the 128-bit (SSE2) path: natural-text words are short (~5 B), so
// AVX2's 32-B stride overshoots and its per-step overhead loses to 16-B on real
// corpora (measured: SSE2 ALL 52.6 vs AVX2 51.7 e2e; prose 52.3 vs 46.4 — see
// X86_PORT_LOG.md). AVX2 stays available behind -DUSE_AVX2_LET for re-measuring.
#if defined(__AVX2__) && defined(USE_AVX2_LET) && !defined(NO_SIMD_LET)
    const __m256i v20 = _mm256_set1_epi8(0x20), va = _mm256_set1_epi8('a');
    const __m256i v25 = _mm256_set1_epi8(25),   vz = _mm256_setzero_si256();
    while (q + 32 <= len) {
        __m256i v   = _mm256_loadu_si256((const __m256i*)(t + q));
        __m256i nl  = _mm256_subs_epu8(_mm256_sub_epi8(_mm256_or_si256(v, v20), va), v25); // 0 iff a-z
        unsigned m  = (unsigned)_mm256_movemask_epi8(_mm256_cmpeq_epi8(nl, vz));           // bit set iff letter
        if (m != 0xFFFFFFFFu) return q + (uint32_t)__builtin_ctz(~m);                      // first non-letter
        q += 32;
    }
#elif defined(__SSE2__) && !defined(NO_SIMD_LET)
    const __m128i v20 = _mm_set1_epi8(0x20), va = _mm_set1_epi8('a');
    const __m128i v25 = _mm_set1_epi8(25),   vz = _mm_setzero_si128();
    while (q + 16 <= len) {
        __m128i v   = _mm_loadu_si128((const __m128i*)(t + q));
        __m128i nl  = _mm_subs_epu8(_mm_sub_epi8(_mm_or_si128(v, v20), va), v25);
        unsigned m  = (unsigned)_mm_movemask_epi8(_mm_cmpeq_epi8(nl, vz)) & 0xFFFFu;
        if (m != 0xFFFFu) return q + (uint32_t)__builtin_ctz((~m) & 0xFFFFu);
        q += 16;
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && !defined(NO_SIMD_LET) && !defined(NO_NEON_LET)
    while (q + 16 <= len) {
        uint8x16_t v = vld1q_u8(t + q);
        uint8x16_t lo = vorrq_u8(v, vdupq_n_u8(0x20));               // ASCII-lowercase fold
        uint8x16_t notlet = vcgtq_u8(vsubq_u8(lo, vdupq_n_u8('a')), vdupq_n_u8(25)); // (lo-'a')>25 → not a-z (also flags >=128)
        if (vmaxvq_u8(notlet)) { uint8_t m[16]; vst1q_u8(m, notlet);
            for (int j=0;j<16;j++) if (m[j]) return q+j; }
        q += 16;
    }
#endif
    while (q < len) { uint8_t b=t[q]; uint8_t l=b|0x20; if ((uint8_t)(l-'a') <= 25) q++; else break; }
    return q;
}

// Advance over a run of ASCII whitespace (\t\n\v\f\r and space) from q, updating
// *lastnl to the last \r/\n position seen. Stops at the first byte that is not ASCII
// whitespace (incl any >=0x80 — caller's scalar loop handles Unicode \s). Predicate
// matches uniclass \s for ASCII exactly: b==0x20 || b in 0x09..0x0D. x86 SSE2; scalar
// elsewhere (M1 keeps its scalar scan unchanged).
static inline uint32_t ascii_ws_run(const uint8_t* t, uint32_t q, uint32_t len, uint32_t* lastnl) {
#if defined(__SSE2__) && !defined(NO_SIMD_LET) && !defined(NO_SIMD_WS)
    const __m128i sp=_mm_set1_epi8(0x20), n9=_mm_set1_epi8(9), four=_mm_set1_epi8(4), z=_mm_setzero_si128();
    const __m128i lf=_mm_set1_epi8(0x0A), cr=_mm_set1_epi8(0x0D);
    while (q + 16 <= len) {
        __m128i v = _mm_loadu_si128((const __m128i*)(t+q));
        __m128i ctl = _mm_cmpeq_epi8(_mm_subs_epu8(_mm_sub_epi8(v,n9),four), z);   // v in 9..13
        __m128i isws = _mm_or_si128(_mm_cmpeq_epi8(v,sp), ctl);
        unsigned mws = (unsigned)_mm_movemask_epi8(isws) & 0xFFFFu;
        unsigned mnl = (unsigned)_mm_movemask_epi8(_mm_or_si128(_mm_cmpeq_epi8(v,lf),_mm_cmpeq_epi8(v,cr))) & 0xFFFFu;
        if (mws != 0xFFFFu) {                                   // run ends in this block
            unsigned stop = __builtin_ctz(~mws & 0xFFFFu);
            unsigned before = mnl & ((1u<<stop)-1);
            if (before) *lastnl = q + 31 - __builtin_clz(before);
            return q + stop;
        }
        if (mnl) *lastnl = q + 31 - __builtin_clz(mnl);
        q += 16;
    }
#endif
    while (q < len) { uint8_t b=t[q]; if (b==0x20 || (uint8_t)(b-9)<=4) { if(b==0x0A||b==0x0D)*lastnl=q; q++; } else break; }
    return q;
}

// Advance over a run of ASCII bytes that are NOT \s, \p{L}, or \p{N} (alt4's class).
// Stops at ws / letter / digit / any byte >=0x80 (Unicode handled by caller's scalar).
static inline uint32_t ascii_punct_run(const uint8_t* t, uint32_t q, uint32_t len) {
#if defined(__SSE2__) && !defined(NO_SIMD_LET) && !defined(NO_SIMD_WS)
    const __m128i n9=_mm_set1_epi8(9), four=_mm_set1_epi8(4), z=_mm_setzero_si128();
    const __m128i sp=_mm_set1_epi8(0x20), v20=_mm_set1_epi8(0x20), va=_mm_set1_epi8('a'), v25=_mm_set1_epi8(25);
    const __m128i v0=_mm_set1_epi8('0'), v9=_mm_set1_epi8(9);
    while (q + 16 <= len) {
        __m128i v = _mm_loadu_si128((const __m128i*)(t+q));
        __m128i ws  = _mm_or_si128(_mm_cmpeq_epi8(v,sp), _mm_cmpeq_epi8(_mm_subs_epu8(_mm_sub_epi8(v,n9),four), z));
        __m128i let = _mm_cmpeq_epi8(_mm_subs_epu8(_mm_sub_epi8(_mm_or_si128(v,v20),va),v25), z);   // (v|0x20) in a..z
        __m128i dig = _mm_cmpeq_epi8(_mm_subs_epu8(_mm_sub_epi8(v,v0),v9), z);                      // v in '0'..'9'
        __m128i hi  = _mm_cmpgt_epi8(z, v);                                                          // v>=0x80 (signed <0)
        unsigned mstop = (unsigned)_mm_movemask_epi8(_mm_or_si128(_mm_or_si128(ws,let),_mm_or_si128(dig,hi))) & 0xFFFFu;
        if (mstop) return q + __builtin_ctz(mstop);
        q += 16;
    }
#endif
    while (q < len) { uint8_t b=t[q]; if (b>=0x80) break; uint8_t l=b|0x20;
        if (b==0x20 || (uint8_t)(b-9)<=4 || (uint8_t)(l-'a')<=25 || (uint8_t)(b-'0')<=9) break; q++; }
    return q;
}

struct UClass {
    std::vector<uint32_t> Llo, Lhi, Nlo, Nhi, Slo, Shi;
    uint8_t a[128];                 // ASCII: bit0=L bit1=N bit2=S
    std::vector<uint8_t> bmp;       // 65536: bits L|N|S for cp<2^16 (O(1), covers CJK & most scripts)
    static bool in(const std::vector<uint32_t>& lo, const std::vector<uint32_t>& hi, uint32_t cp) {
        int a=0, b=(int)lo.size()-1;
        while (a<=b){ int m=(a+b)>>1; if(cp<lo[m]) b=m-1; else if(cp>hi[m]) a=m+1; else return true; }
        return false;
    }
    inline bool isL(uint32_t cp) const { return cp<65536 ? (bmp[cp]&1) : in(Llo,Lhi,cp); }
    inline bool isN(uint32_t cp) const { return cp<65536 ? (bmp[cp]&2) : in(Nlo,Nhi,cp); }
    inline bool isS(uint32_t cp) const { return cp<65536 ? (bmp[cp]&4) : in(Slo,Shi,cp); }
    static void rd(FILE* f, std::vector<uint32_t>& lo, std::vector<uint32_t>& hi, const char* path){
        auto fail = [&]{ fclose(f); throw std::runtime_error(std::string("quicktok: bad uniclass file: ") + path); };
        uint32_t n; if(fread(&n,4,1,f)!=1) fail();
        if (n > 100000) fail();                      // sane range-count bound
        lo.resize(n); hi.resize(n);
        for(uint32_t i=0;i<n;i++){ uint32_t v[2]; if(fread(v,4,2,f)!=2) fail();
            if (v[0] > v[1] || v[1] > 0x10FFFF) fail();
            lo[i]=v[0]; hi[i]=v[1]; }
    }
    static UClass load(const char* path){
        FILE* f=fopen(path,"rb");
        if(!f) throw std::runtime_error(std::string("quicktok: cannot open uniclass file: ") + path);
        UClass U; rd(f,U.Llo,U.Lhi,path); rd(f,U.Nlo,U.Nhi,path); rd(f,U.Slo,U.Shi,path); fclose(f);
        U.bmp.assign(65536, 0);
        auto mark=[&](std::vector<uint32_t>&lo,std::vector<uint32_t>&hi,uint8_t bit){ for(size_t i=0;i<lo.size();i++){ uint32_t a=lo[i],b=hi[i]>65535?65535:hi[i]; if(lo[i]<65536) for(uint32_t c=a;c<=b;c++) U.bmp[c]|=bit; } };
        mark(U.Llo,U.Lhi,1); mark(U.Nlo,U.Nhi,2); mark(U.Slo,U.Shi,4);
        for(uint32_t c=0;c<128;c++) U.a[c]=U.bmp[c];
        return U;
    }
};

// decode one UTF-8 codepoint at p; returns codepoint, sets *nb to byte length (1..4)
static inline uint32_t u8dec(const uint8_t* t, uint32_t p, uint32_t len, uint32_t* nb) {
    uint8_t c = t[p];
    if (c < 0x80) { *nb=1; return c; }
    if ((c>>5)==0x6 && p+1<len) { *nb=2; return ((c&0x1F)<<6)|(t[p+1]&0x3F); }
    if ((c>>4)==0xE && p+2<len) { *nb=3; return ((c&0x0F)<<12)|((t[p+1]&0x3F)<<6)|(t[p+2]&0x3F); }
    if ((c>>3)==0x1E && p+3<len) { *nb=4; return ((uint32_t)(c&0x07)<<18)|((t[p+1]&0x3F)<<12)|((t[p+2]&0x3F)<<6)|(t[p+3]&0x3F); }
    *nb=1; return c;  // malformed: treat as 1 byte
}

// Find ONE pretoken starting at p; returns its byte length (no emit/advance).
// Exposed so e2e can fuse pretok with merge. (Word fast-path lives in the e2e
// fused loop; here we go straight to the alt cascade for generality.)
//
// Two axes parameterize the three cl100k-family grammars we support — letters,
// contractions ('s 't 're 've 'm 'll 'd) and punct (alt4) are identical across
// all three; only these differ:
//   O200K_WS    : whitespace cascade is the o200k-style \s*[\r\n]+ | \s+(?!\S) | \s+
//                 (Llama-3, Qwen) vs cl100k's \s+$ | \s*[\r\n] | \s+(?!\S) | \s.
//   SINGLE_DIGIT: number alt is \p{N} (one digit per token — Qwen) vs \p{N}{1,3}.
template <bool O200K_WS, bool SINGLE_DIGIT>
static inline uint32_t pretok_next_impl(const UClass& U, const uint8_t* t, uint32_t p, uint32_t len) {
    uint8_t b = t[p];
    uint32_t nb; uint32_t cp = u8dec(t, p, len, &nb);
    // --- alt 1: '(?i:[sdmt]|ll|ve|re) ---
    if (b == '\'') {
        auto lc = [](uint8_t x){ return (uint8_t)(x>='A'&&x<='Z'? x+32 : x); };
        if (p+1 < len) {
            uint8_t c1 = lc(t[p+1]);
            if (c1=='s'||c1=='d'||c1=='m'||c1=='t') return 2;
            if (p+2 < len) { uint8_t c2 = lc(t[p+2]);
                if ((c1=='l'&&c2=='l')||(c1=='v'&&c2=='e')||(c1=='r'&&c2=='e')) return 3; }
        }
    }
    // --- alt 2: [^\r\n\p{L}\p{N}]?+ \p{L}++ ---
    {
        uint32_t q = p; bool pfx = false;
        bool cpL = U.isL(cp);
        if (!cpL && cp!='\r' && cp!='\n' && !U.isN(cp) && p+nb < len) {
            uint32_t nb2; uint32_t c2 = u8dec(t, p+nb, len, &nb2);   // p+nb<len: in-bounds
            if (U.isL(c2)) { pfx = true; q = p+nb; }
        }
        if (pfx || cpL) {
            if (!pfx) q = p;
            for (;;) {
                q = ascii_letter_run(t, q, len);
                if (q >= len || t[q] < 0x80) break;
                uint32_t n2; uint32_t c2=u8dec(t,q,len,&n2); if (U.isL(c2)) q+=n2; else break;
            }
            return q - p;
        }
    }
    // --- alt 3: \p{N}{1,3}+  (Qwen: \p{N} — exactly one digit codepoint) ---
    if (U.isN(cp)) {
        if constexpr (SINGLE_DIGIT) return nb;   // one \p{N} codepoint = nb bytes
        uint32_t q=p, cnt=0;
        while (q<len && cnt<3) { uint32_t n2; uint32_t c2=u8dec(t,q,len,&n2); if(U.isN(c2)){q+=n2;cnt++;} else break; }
        return q - p;
    }
    // --- alt 4: ' ?[^\s\p{L}\p{N}]++[\r\n]*+ ---
    {
        uint32_t q = p; if (b==' ') q+=1; uint32_t s4 = q;
        q = ascii_punct_run(t, q, len);                                 // ASCII punct/symbol run (SIMD)
        while (q<len){ uint32_t n2; uint32_t c2=u8dec(t,q,len,&n2); if(!U.isS(c2)&&!U.isL(c2)&&!U.isN(c2)){q+=n2;} else break; }  // Unicode tail
        if (q > s4) { while (q<len && (t[q]=='\r'||t[q]=='\n')) q+=1; return q - p; }
    }
    // --- whitespace alts (5,6,7,8) on the maximal \s run [p,e) ---
    {
        uint32_t e=p; uint32_t lastnl=UINT32_MAX; uint32_t lastlen=1;
        uint32_t e0 = ascii_ws_run(t, e, len, &lastnl); if (e0 > e) { lastlen = 1; e = e0; }   // ASCII \s run (SIMD)
        while (e<len){ uint32_t n2; uint32_t c2=u8dec(t,e,len,&n2); if(!U.isS(c2)) break; if(c2=='\r'||c2=='\n') lastnl=e; lastlen=n2; e+=n2; }  // Unicode \s tail
        if (e==p) return 1;
        if constexpr (O200K_WS) {  // Llama-3 / Qwen whitespace: \s*[\r\n]+ | \s+(?!\S) | \s+
            if (lastnl != UINT32_MAX) return lastnl+1 - p;     // \s*[\r\n]+
            if (e == len)            return e - p;              // \s+(?!\S), run to EOF
            if (e - p > lastlen)     return (e-lastlen) - p;    // \s+(?!\S)
            return e - p;                                       // \s+
        } else {                  // cl100k: \s++$ | \s*[\r\n] | \s+(?!\S) | \s
            if (e==len) return e - p;                          // alt5
            if (lastnl != UINT32_MAX) return lastnl+1 - p;     // alt6
            if (e - p > lastlen) return (e-lastlen) - p;       // alt7
            return nb;                                         // alt8
        }
    }
}

static inline uint32_t pretok_next(const UClass& U, const uint8_t* t, uint32_t p, uint32_t len) {
    return pretok_next_impl<false, false>(U, t, p, len);   // cl100k
}
static inline uint32_t pretok_next_llama3(const UClass& U, const uint8_t* t, uint32_t p, uint32_t len) {
    return pretok_next_impl<true, false>(U, t, p, len);    // Llama-3 (cl100k grammar + o200k-style \s)
}
static inline uint32_t pretok_next_qwen(const UClass& U, const uint8_t* t, uint32_t p, uint32_t len) {
    return pretok_next_impl<true, true>(U, t, p, len);     // Qwen2.5/Qwen3 (o200k-style \s + single-digit \p{N})
}

template <class CB>
static void pretok(const UClass& U, const uint8_t* t, uint32_t len, CB&& cb) {
    uint32_t p = 0;
    while (p < len) {
        // word fast path (kept here so the merge-only pretok bench stays fast)
        uint8_t b = t[p]; uint32_t ls = (b==' ') ? p+1 : p;
        if (ls < len && (uint8_t)((t[ls]|0x20)-'a') <= 25u) {
            uint32_t q = ascii_letter_run(t, ls, len);
            if (q==len || t[q] < 0x80) { cb(p, q-p); p=q; continue; }
        }
        uint32_t l = pretok_next(U, t, p, len);
        cb(p, l); p += l;
    }
}

}  // namespace detail
}  // namespace quicktok
