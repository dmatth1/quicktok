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
#include <vector>
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <immintrin.h>
#endif

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
    static void rd(FILE* f, std::vector<uint32_t>& lo, std::vector<uint32_t>& hi){
        uint32_t n; if(fread(&n,4,1,f)!=1) exit(1); lo.resize(n); hi.resize(n);
        for(uint32_t i=0;i<n;i++){ uint32_t v[2]; if(fread(v,4,2,f)!=2) exit(1); lo[i]=v[0]; hi[i]=v[1]; }
    }
    static UClass load(const char* path){
        FILE* f=fopen(path,"rb"); if(!f){fprintf(stderr,"uniclass %s\n",path);exit(1);}
        UClass U; rd(f,U.Llo,U.Lhi); rd(f,U.Nlo,U.Nhi); rd(f,U.Slo,U.Shi); fclose(f);
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
static inline uint32_t pretok_next(const UClass& U, const uint8_t* t, uint32_t p, uint32_t len) {
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
        if (!cpL && cp!='\r' && cp!='\n' && !U.isN(cp)) {
            uint32_t nb2; uint32_t c2 = u8dec(t, p+nb, len, &nb2);
            if (p+nb < len && U.isL(c2)) { pfx = true; q = p+nb; }
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
    // --- alt 3: \p{N}{1,3}+ ---
    if (U.isN(cp)) {
        uint32_t q=p, cnt=0;
        while (q<len && cnt<3) { uint32_t n2; uint32_t c2=u8dec(t,q,len,&n2); if(U.isN(c2)){q+=n2;cnt++;} else break; }
        return q - p;
    }
    // --- alt 4: ' ?[^\s\p{L}\p{N}]++[\r\n]*+ ---
    {
        uint32_t q = p; if (b==' ') q+=1;
        uint32_t cnt=0;
        while (q<len){ uint32_t n2; uint32_t c2=u8dec(t,q,len,&n2); if(!U.isS(c2)&&!U.isL(c2)&&!U.isN(c2)){q+=n2;cnt++;} else break; }
        if (cnt>=1) { while (q<len && (t[q]=='\r'||t[q]=='\n')) q+=1; return q - p; }
    }
    // --- whitespace alts (5,6,7,8) on the maximal \s run [p,e) ---
    {
        uint32_t e=p; uint32_t lastnl=UINT32_MAX; uint32_t lastlen=1;
        while (e<len){ uint32_t n2; uint32_t c2=u8dec(t,e,len,&n2); if(!U.isS(c2)) break; if(c2=='\r'||c2=='\n') lastnl=e; lastlen=n2; e+=n2; }
        if (e==p) return 1;
        if (e==len) return e - p;                              // alt5
        if (lastnl != UINT32_MAX) return lastnl+1 - p;         // alt6
        if (e - p > lastlen) return (e-lastlen) - p;           // alt7
        return nb;                                             // alt8
    }
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
