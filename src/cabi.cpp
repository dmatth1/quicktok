// C ABI implementation — thin exception-catching wrapper over the C++ API.
#include "quicktok.h"
#include "quicktok.hpp"
#include <cstring>
#include <new>

struct qt_tokenizer {
    quicktok::Tokenizer tok;
    explicit qt_tokenizer(quicktok::Tokenizer&& t) : tok(std::move(t)) {}
};

static void put_err(char* buf, size_t cap, const char* msg) {
    if (!buf || !cap) return;
    size_t n = strlen(msg);
    if (n >= cap) n = cap - 1;
    memcpy(buf, msg, n);
    buf[n] = 0;
}

extern "C" {

qt_tokenizer* qt_load_dir(const char* dir, const char* encoding,
                          char* errbuf, size_t errbuf_len) {
    try {
        auto t = quicktok::Tokenizer::load_dir(dir ? dir : "",
                                               encoding ? encoding : "cl100k_base");
        return new qt_tokenizer(std::move(t));
    } catch (const std::exception& e) {
        put_err(errbuf, errbuf_len, e.what());
        return nullptr;
    } catch (...) {
        put_err(errbuf, errbuf_len, "quicktok: unknown error");
        return nullptr;
    }
}

void qt_tokenizer_free(qt_tokenizer* t) { delete t; }

static int encode_impl(const qt_tokenizer* t, const char* text, size_t len,
                       uint32_t** ids, size_t* n, bool special) {
    if (!t || !ids || !n || (!text && len)) return 1;
    *ids = nullptr; *n = 0;
    try {
        std::vector<uint32_t> v = special
            ? t->tok.encode_with_special(std::string_view(text, len))
            : t->tok.encode(std::string_view(text, len));
        uint32_t* out = new uint32_t[v.size() ? v.size() : 1];
        memcpy(out, v.data(), v.size() * sizeof(uint32_t));
        *ids = out; *n = v.size();
        return 0;
    } catch (...) { return 1; }
}

int qt_encode(const qt_tokenizer* t, const char* text, size_t len,
              uint32_t** ids, size_t* n) {
    return encode_impl(t, text, len, ids, n, false);
}
int qt_encode_with_special(const qt_tokenizer* t, const char* text, size_t len,
                           uint32_t** ids, size_t* n) {
    return encode_impl(t, text, len, ids, n, true);
}
void qt_ids_free(uint32_t* ids) { delete[] ids; }

int qt_decode(const qt_tokenizer* t, const uint32_t* ids, size_t n,
              char** out, size_t* outlen) {
    if (!t || !out || !outlen || (!ids && n)) return 1;
    *out = nullptr; *outlen = 0;
    try {
        std::string s;
        t->tok.decode(ids, n, s);
        char* o = new char[s.size() + 1];
        memcpy(o, s.data(), s.size());
        o[s.size()] = 0;
        *out = o; *outlen = s.size();
        return 0;
    } catch (...) { return 1; }
}
void qt_str_free(char* s) { delete[] s; }

size_t qt_count(const qt_tokenizer* t, const char* text, size_t len) {
    if (!t || (!text && len)) return (size_t)-1;
    try { return t->tok.count(std::string_view(text, len)); }
    catch (...) { return (size_t)-1; }
}

size_t qt_vocab_size(const qt_tokenizer* t) { return t ? t->tok.vocab_size() : 0; }
const char* qt_encoding(const qt_tokenizer* t) { return t ? t->tok.encoding().c_str() : ""; }

}  // extern "C"
