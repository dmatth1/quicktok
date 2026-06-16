// Unicode NFC normalization (UAX #15 canonical composition) for encodings whose
// reference pipeline normalizes before tokenizing (Qwen: tokenizer.json carries
// "normalizer": {"type": "NFC"}).
//
// Fast path: clean() scans for "suspicious" codepoints — anything that could
// make a string non-NFC (see tools/export_nfc.py for the exact contract). Real
// text is almost always clean (ASCII and CJK never hit it; precomposed Latin
// accents don't either), so encode pays one cheap scan and zero copies; only
// genuinely dirty input (e.g. U+2329, decomposed accents, lone jamo) takes the
// normalize() slow path. Tables in data/nfc.bin, derived + version-stamped from
// Python's unicodedata (tools/export_nfc.py, `verify` re-derives exhaustively).
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#include "pretok.hpp"   // u8dec

namespace quicktok {
namespace detail {

struct NFC {
    // suspicious set: bitmap for BMP, sorted ranges above
    std::vector<uint8_t> sbmp;                 // 65536 bits -> 8KB
    std::vector<uint32_t> slo, shi;
    // ccc / decomp / composition (sorted, binary-searched; all tiny)
    std::vector<uint32_t> ccc_cp; std::vector<uint8_t> ccc_v;
    std::vector<uint32_t> dcp, doff;           // decomp: cp -> [doff[i], doff[i+1]) in dpool
    std::vector<uint32_t> dpool;
    std::vector<uint64_t> comp_k;              // (a<<21)|b, sorted
    std::vector<uint32_t> comp_v;
    bool loaded = false;

    static constexpr uint32_t SB = 0xAC00, LB = 0x1100, VB = 0x1161, TB = 0x11A7;
    static constexpr uint32_t LC = 19, VC = 21, TC = 28, NC = VC * TC, SC = LC * NC;

    inline bool suspicious(uint32_t cp) const {
        if (cp < 0x80) return false;
        if (cp < 65536) return (sbmp[cp >> 3] >> (cp & 7)) & 1;
        size_t i = std::upper_bound(slo.begin(), slo.end(), cp) - slo.begin();
        return i > 0 && cp <= shi[i - 1];
    }
    inline uint8_t ccc(uint32_t cp) const {
        auto it = std::lower_bound(ccc_cp.begin(), ccc_cp.end(), cp);
        return (it != ccc_cp.end() && *it == cp) ? ccc_v[it - ccc_cp.begin()] : 0;
    }
    inline uint32_t compose(uint32_t a, uint32_t b) const {
        // Hangul: L+V -> LV, LV+T -> LVT (algorithmic)
        if (a >= LB && a < LB + LC && b >= VB && b < VB + VC)
            return SB + ((a - LB) * VC + (b - VB)) * TC;
        if (a >= SB && a < SB + SC && (a - SB) % TC == 0 && b > TB && b <= TB + TC - 1)
            return a + (b - TB);
        uint64_t k = ((uint64_t)a << 21) | b;
        auto it = std::lower_bound(comp_k.begin(), comp_k.end(), k);
        return (it != comp_k.end() && *it == k) ? comp_v[it - comp_k.begin()] : 0;
    }

    // true iff no suspicious codepoint (=> the input is already NFC)
    bool clean(const uint8_t* t, size_t n) const {
        size_t i = 0;
        while (i < n) {
            // ASCII run skip, 8 bytes at a time
            while (i + 8 <= n) {
                uint64_t w; std::memcpy(&w, t + i, 8);
                if (w & 0x8080808080808080ull) break;
                i += 8;
            }
            while (i < n && t[i] < 0x80) i++;
            if (i >= n) break;
            uint32_t nb, cp = u8dec(t, (uint32_t)i, (uint32_t)n, &nb);
            if (suspicious(cp)) return false;
            i += nb;
        }
        return true;
    }

    // NFC of [t, t+n) appended to out. Clean spans are copied verbatim; full
    // normalization runs only on dirty WINDOWS — from the clean codepoint
    // preceding a suspicious one (it may absorb a combining mark, e.g. e+U+0301)
    // to the next clean codepoint (every non-suspicious cp has ccc==0, so it is
    // a starter and seals the window). Cost is ~memcpy + O(dirty bytes), not
    // O(n) normalization — one stray U+2329 in 25 MB must not cost a full pass.
    void normalize(const uint8_t* t, size_t n, std::string& out) const {
        size_t i = 0, copied = 0, last_clean = SIZE_MAX;
        while (i < n) {
            if (t[i] < 0x80) {
                while (i + 8 <= n) {                       // ASCII run skip
                    uint64_t w; std::memcpy(&w, t + i, 8);
                    if (w & 0x8080808080808080ull) break;
                    i += 8;
                }
                while (i < n && t[i] < 0x80) i++;
                last_clean = i - 1;                        // ASCII is always clean (1-byte cp)
                continue;
            }
            uint32_t nb, cp = u8dec(t, (uint32_t)i, (uint32_t)n, &nb);
            if (!suspicious(cp)) { last_clean = i; i += nb; continue; }
            size_t ws = (last_clean != SIZE_MAX && last_clean >= copied) ? last_clean : i;
            size_t j = i + nb;                             // extend to next clean cp
            while (j < n) {
                if (t[j] < 0x80) break;
                uint32_t nb2, cp2 = u8dec(t, (uint32_t)j, (uint32_t)n, &nb2);
                if (!suspicious(cp2)) break;
                j += nb2;
            }
            out.append((const char*)t + copied, ws - copied);
            nfc_window(t + ws, j - ws, out);
            copied = j; i = j; last_clean = SIZE_MAX;
        }
        out.append((const char*)t + copied, n - copied);
    }

    // full NFC of one dirty window (decompose -> reorder -> compose); windows are tiny.
    // Malformed UTF-8 bytes inside a window are carried through VERBATIM via a
    // sentinel (RAWBIT | byte): they act as starters (ccc 0, never compose) and are
    // re-emitted as the original byte — normalization must not "repair" invalid input.
    static constexpr uint32_t RAWBIT = 0x80000000u;
    void nfc_window(const uint8_t* t, size_t n, std::string& out) const {
        std::vector<uint32_t> buf;
        buf.reserve(n);
        size_t i = 0;
        while (i < n) {                                   // canonical decomposition
            uint32_t nb, cp = u8dec(t, (uint32_t)i, (uint32_t)n, &nb);
            if (cp >= 0x80 && nb == 1) {                  // malformed byte: passthrough
                buf.push_back(RAWBIT | t[i]); i += 1; continue;
            }
            i += nb;
            if (cp >= SB && cp < SB + SC) {               // Hangul: algorithmic
                uint32_t s = cp - SB;
                buf.push_back(LB + s / NC);
                buf.push_back(VB + (s % NC) / TC);
                if (s % TC) buf.push_back(TB + s % TC);
                continue;
            }
            auto it = std::lower_bound(dcp.begin(), dcp.end(), cp);
            if (it != dcp.end() && *it == cp) {
                size_t k = it - dcp.begin();
                for (uint32_t j = doff[k]; j < doff[k + 1]; j++) buf.push_back(dpool[j]);
            } else {
                buf.push_back(cp);
            }
        }
        for (size_t s = 0; s < buf.size(); ) {            // canonical reordering
            if (ccc(buf[s]) == 0) { s++; continue; }
            size_t e = s;
            while (e < buf.size() && ccc(buf[e]) != 0) e++;
            for (size_t a = s + 1; a < e; a++)            // insertion sort (runs are tiny), stable
                for (size_t b = a; b > s && ccc(buf[b - 1]) > ccc(buf[b]); b--)
                    std::swap(buf[b - 1], buf[b]);
            s = e;
        }
        std::vector<uint32_t> res;                        // canonical composition
        res.reserve(buf.size());
        size_t starter = SIZE_MAX;
        uint8_t lastcc = 0;
        for (uint32_t cp : buf) {
            uint8_t cc = ccc(cp);
            if (starter != SIZE_MAX && (starter == res.size() - 1 || lastcc < cc)) {
                if (uint32_t m = compose(res[starter], cp)) { res[starter] = m; continue; }
            }
            if (cc == 0) { starter = res.size(); lastcc = 0; }
            else lastcc = cc;
            res.push_back(cp);
        }
        for (uint32_t cp : res) {                         // encode UTF-8
            if (cp & RAWBIT) { out += (char)(cp & 0xFF); continue; }  // verbatim malformed byte
            if (cp < 0x80) out += (char)cp;
            else if (cp < 0x800) { out += (char)(0xC0 | (cp >> 6)); out += (char)(0x80 | (cp & 0x3F)); }
            else if (cp < 0x10000) { out += (char)(0xE0 | (cp >> 12)); out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F)); }
            else { out += (char)(0xF0 | (cp >> 18)); out += (char)(0x80 | ((cp >> 12) & 0x3F)); out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F)); }
        }
    }

    static NFC load(const char* path) {
        FILE* f = fopen(path, "rb");
        if (!f) throw std::runtime_error(std::string("quicktok: cannot open nfc table: ") + path);
        auto fail = [&] { fclose(f); throw std::runtime_error(std::string("quicktok: bad nfc table: ") + path); };
        auto rd32 = [&]() -> uint32_t { uint32_t v; if (fread(&v, 4, 1, f) != 1) fail(); return v; };
        NFC N;
        N.sbmp.assign(8192, 0);
        uint32_t nr = rd32(); if (nr > 100000) fail();
        for (uint32_t i = 0; i < nr; i++) {
            uint32_t lo = rd32(), hi = rd32();
            if (lo > hi || hi > 0x10FFFF) fail();
            for (uint32_t cp = lo; cp <= hi && cp < 65536; cp++) N.sbmp[cp >> 3] |= (uint8_t)(1u << (cp & 7));
            if (hi >= 65536) { N.slo.push_back(std::max(lo, 65536u)); N.shi.push_back(hi); }
        }
        uint32_t nc = rd32(); if (nc > 100000) fail();
        N.ccc_cp.resize(nc); N.ccc_v.resize(nc);
        for (uint32_t i = 0; i < nc; i++) { N.ccc_cp[i] = rd32(); N.ccc_v[i] = (uint8_t)rd32(); }
        uint32_t nd = rd32(); if (nd > 100000) fail();
        N.dcp.resize(nd); N.doff.resize(nd + 1, 0);
        for (uint32_t i = 0; i < nd; i++) {
            N.dcp[i] = rd32();
            uint32_t len = rd32(); if (len == 0 || len > 8) fail();
            N.doff[i + 1] = N.doff[i] + len;
            for (uint32_t j = 0; j < len; j++) N.dpool.push_back(rd32());
        }
        uint32_t np = rd32(); if (np > 100000) fail();
        N.comp_k.resize(np); N.comp_v.resize(np);
        for (uint32_t i = 0; i < np; i++) {
            uint32_t a = rd32(), b = rd32();
            N.comp_k[i] = ((uint64_t)a << 21) | b;
            N.comp_v[i] = rd32();
        }
        fclose(f);
        N.loaded = true;
        return N;
    }
};

}  // namespace detail
}  // namespace quicktok
