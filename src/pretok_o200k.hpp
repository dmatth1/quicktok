// Hand-coded pretokenizer for the FIXED o200k_base regex (GPT-4o). Reproduces the
// 7 alternatives (ordered, first-match, greedy):
//   1: [^\r\n\p{L}\p{N}]? [UPPER]* [LOWER]+ (?i:'s|'t|'re|'ve|'m|'ll|'d)?
//   2: [^\r\n\p{L}\p{N}]? [UPPER]+ [LOWER]* (?i:'s|'t|'re|'ve|'m|'ll|'d)?
//   3: \p{N}{1,3}
//   4:  ?[^\s\p{L}\p{N}]+ [\r\n/]*
//   5: \s*[\r\n]+      6: \s+(?!\S)      7: \s+
// where UPPER=[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}], LOWER=[\p{Ll}\p{Lm}\p{Lo}\p{M}].
// Classes from fixtures/uniclass_o200k.bin (L,N,S,UPPER,LOWER). Exact-verified vs
// tiktoken's o200k split (pretok_o200k_test.cpp).
#pragma once
#include "pretok.hpp"   // u8dec

namespace quicktok {
namespace detail {

struct UClassO {
    // 5 classes: 0=L 1=N 2=S 3=UPPER 4=LOWER
    std::vector<uint32_t> lo[5], hi[5];
    std::vector<uint8_t> bmp;   // 65536, bit c = class c
    static bool rin(const std::vector<uint32_t>& lo, const std::vector<uint32_t>& hi, uint32_t cp) {
        int a=0,b=(int)lo.size()-1; while(a<=b){int m=(a+b)>>1; if(cp<lo[m])b=m-1; else if(cp>hi[m])a=m+1; else return true;} return false;
    }
    inline bool cls(int c, uint32_t cp) const { return cp<65536 ? ((bmp[cp]>>c)&1) : rin(lo[c],hi[c],cp); }
    inline bool isL(uint32_t cp) const { return cls(0,cp); }
    inline bool isN(uint32_t cp) const { return cls(1,cp); }
    inline bool isS(uint32_t cp) const { return cls(2,cp); }
    inline bool isU(uint32_t cp) const { return cls(3,cp); }   // UPPER class
    inline bool isLo(uint32_t cp) const { return cls(4,cp); }  // LOWER class
    static UClassO load(const char* path) {
        FILE* f=fopen(path,"rb");
        if(!f) throw std::runtime_error(std::string("quicktok: cannot open uniclass file: ") + path);
        auto fail = [&]{ fclose(f); throw std::runtime_error(std::string("quicktok: bad uniclass file: ") + path); };
        UClassO U; U.bmp.assign(65536,0);
        for (int c=0;c<5;c++){ uint32_t n; if(fread(&n,4,1,f)!=1) fail();
            if (n > 100000) fail();
            U.lo[c].resize(n); U.hi[c].resize(n);
            for(uint32_t i=0;i<n;i++){ uint32_t v[2]; if(fread(v,4,2,f)!=2) fail();
                if (v[0] > v[1] || v[1] > 0x10FFFF) fail();
                U.lo[c][i]=v[0]; U.hi[c][i]=v[1];
                uint32_t a=v[0], b=v[1]>65535?65535:v[1]; if(v[0]<65536) for(uint32_t cp=a;cp<=b;cp++) U.bmp[cp]|=(uint8_t)(1<<c); } }
        fclose(f); return U;
    }
};

// alt1: [UPPER]*[LOWER]+ at start -> end, or 0. Greedy U* then L+.
// ASCII fast paths: for cp < 0x80, UPPER == [A-Z] and LOWER == [a-z] exactly
// (ASCII has no Lt/Lm/Lo/M), so the SIMD case runs are the same predicate as the
// scalar u8dec+cls loop — Unicode chars fall through to the original scalar steps.
static inline uint32_t o_matchUL(const UClassO& U, const uint8_t* t, uint32_t start, uint32_t len) {
    uint32_t i=start, n; bool sawmb=false;
    for (;;) {                                                  // UPPER* -> [start,i)
        i = ascii_upper_run(t, i, len);
        if (i >= len || t[i] < 0x80) break;
        uint32_t cp=u8dec(t,i,len,&n); if (U.isU(cp)) { i+=n; sawmb=true; } else break;
    }
    uint32_t j=i; bool any=false;
    for (;;) {                                                  // LOWER+
        uint32_t j2 = ascii_lower_run(t, j, len); if (j2 > j) { j = j2; any = true; }
        if (j >= len || t[j] < 0x80) break;
        uint32_t cp=u8dec(t,j,len,&n); if (U.isLo(cp)) { j+=n; any=true; } else break;
    }
    if (any) return j;
    // LOWER+ empty: backtrack the greedy UPPER* to the LAST LOWER-eligible (BOTH) char in
    // [start,i) — it becomes the trailing LOWER+. The whole UPPER* run is LOWER-eligible up
    // to that point, so the match ends right after it. (e.g. "亚洲AV": UPPER* grabbed 亚洲AV,
    // but 亚洲 are Lo/BOTH and AV are Lu/UPPER-only, so alt1 matches "亚洲", not "亚洲AV".)
    // A pure-ASCII UPPER* run ([A-Z] only, sawmb=false) has no LOWER-eligible char: skip the scan.
    if (!sawmb) return 0;
    uint32_t e=0, p=start;
    while (p<i) { uint32_t cp=u8dec(t,p,len,&n); if (U.isLo(cp)) e=p+n; p+=n; }
    return e;   // 0 if the UPPER* run has no LOWER-eligible char (alt1 truly fails -> alt2)
}
// alt2: [UPPER]+[LOWER]* at start -> end, or 0.
static inline uint32_t o_matchUpL(const UClassO& U, const uint8_t* t, uint32_t start, uint32_t len) {
    uint32_t i=start, n;
    for (;;) {                                                  // UPPER+
        i = ascii_upper_run(t, i, len);
        if (i >= len || t[i] < 0x80) break;
        uint32_t cp=u8dec(t,i,len,&n); if (U.isU(cp)) i+=n; else break;
    }
    if (i == start) return 0;
    uint32_t j=i;
    for (;;) {                                                  // LOWER*
        j = ascii_lower_run(t, j, len);
        if (j >= len || t[j] < 0x80) break;
        uint32_t cp=u8dec(t,j,len,&n); if (U.isLo(cp)) j+=n; else break;
    }
    return j;
}
// (?i:'s|'t|'re|'ve|'m|'ll|'d)? suffix after a letter match ending at e.
static inline uint32_t o_contraction(const uint8_t* t, uint32_t e, uint32_t len) {
    if (e>=len || t[e] != '\'') return e;
    auto lc=[](uint8_t x){ return (uint8_t)(x>='A'&&x<='Z'? x+32 : x); };
    if (e+1>=len) return e;
    uint8_t c1=lc(t[e+1]);
    if (c1=='s'||c1=='t'||c1=='m'||c1=='d') return e+2;
    if (e+2<len){ uint8_t c2=lc(t[e+2]); if ((c1=='r'&&c2=='e')||(c1=='v'&&c2=='e')||(c1=='l'&&c2=='l')) return e+3; }
    return e;
}

// Find ONE o200k-family pretoken starting at p; returns its byte length.
// Two parametric axes cover the family's known variants:
//   CONTR        : alts 1&2 carry the (?i:'s|'t|'re|'ve|'m|'ll|'d)? suffix
//                  (o200k/Llama-4: yes; Mistral Tekken: no)
//   SINGLE_DIGIT : alt 3 is \p{N} (Tekken) vs \p{N}{1,3} (o200k)
template <bool CONTR, bool SINGLE_DIGIT>
static inline uint32_t pretok_next_o200k_impl(const UClassO& U, const uint8_t* t, uint32_t p, uint32_t len) {
    uint8_t b = t[p];
    uint32_t nb; uint32_t cp = u8dec(t, p, len, &nb);
    // --- alts 1 & 2 (letters), each tried prefix-consumed-first then prefix-empty ---
    bool prefelig = (cp!='\r' && cp!='\n' && !U.isL(cp) && !U.isN(cp));
    for (int alt=0; alt<2; alt++) {
        uint32_t e = 0;
        if (prefelig) e = alt==0 ? o_matchUL(U,t,p+nb,len) : o_matchUpL(U,t,p+nb,len);   // prefix consumed
        if (e==0)     e = alt==0 ? o_matchUL(U,t,p,len)     : o_matchUpL(U,t,p,len);       // prefix empty
        if (e>0) { if constexpr (CONTR) e = o_contraction(t, e, len); return e - p; }
    }
    // --- alt 3: \p{N}{1,3}  (SINGLE_DIGIT: \p{N}) ---
    if (U.isN(cp)) {
        if constexpr (SINGLE_DIGIT) return nb;
        uint32_t q=p, cnt=0;
        while (q<len && cnt<3) { uint32_t n2; uint32_t c2=u8dec(t,q,len,&n2); if(U.isN(c2)){q+=n2;cnt++;} else break; }
        return q - p;
    }
    // --- alt 4:  ?[^\s\p{L}\p{N}]+[\r\n/]* ---
    {
        uint32_t q = p; if (b==' ') q+=1; uint32_t s4=q;
        q = ascii_punct_run(t, q, len);                                 // ASCII punct/symbol run (SIMD)
        while (q<len){ uint32_t n2; uint32_t c2=u8dec(t,q,len,&n2); if(!U.isS(c2)&&!U.isL(c2)&&!U.isN(c2)){q+=n2;} else break; }
        if (q > s4) { while (q<len && (t[q]=='\r'||t[q]=='\n'||t[q]=='/')) q+=1; return q - p; }
    }
    // --- whitespace alts (5,6,7) on the maximal \s run [p,e) ---
    {
        uint32_t e=p, lastnl=UINT32_MAX, lastlen=1;
        uint32_t e0 = ascii_ws_run(t, e, len, &lastnl); if (e0 > e) { lastlen = 1; e = e0; }   // ASCII \s run (SIMD)
        while (e<len){ uint32_t n2; uint32_t c2=u8dec(t,e,len,&n2); if(!U.isS(c2)) break; if(c2=='\r'||c2=='\n') lastnl=e; lastlen=n2; e+=n2; }
        if (e==p) return 1;                              // not whitespace (shouldn't happen) -> 1 byte
        if (lastnl != UINT32_MAX) return lastnl+1 - p;   // alt5 \s*[\r\n]+
        if (e == len)            return e - p;            // alt6 \s+(?!\S), run to EOF
        if (e - p > lastlen)     return (e-lastlen) - p;  // alt6 \s+(?!\S), all but last ws char
        return e - p;                                     // alt7 \s+ (single ws char before non-ws)
    }
}

static inline uint32_t pretok_next_o200k(const UClassO& U, const uint8_t* t, uint32_t p, uint32_t len) {
    return pretok_next_o200k_impl<true, false>(U, t, p, len);    // o200k / Llama-4
}
static inline uint32_t pretok_next_tekken(const UClassO& U, const uint8_t* t, uint32_t p, uint32_t len) {
    return pretok_next_o200k_impl<false, true>(U, t, p, len);    // Mistral Tekken v3
}

}  // namespace detail
}  // namespace quicktok
