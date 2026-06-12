/* quicktok C ABI — stable FFI surface over the C++ library.
 * All functions are thread-safe on a loaded handle (same guarantee as the C++
 * API). Errors: constructors return NULL and write a message into errbuf;
 * other calls return nonzero / (size_t)-1 on failure. */
#ifndef QUICKTOK_H
#define QUICKTOK_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct qt_tokenizer qt_tokenizer;

/* encoding: "cl100k_base" or "o200k_base"; dir holds the data files.
 * Returns NULL on error and writes a NUL-terminated message into errbuf. */
qt_tokenizer* qt_load_dir(const char* dir, const char* encoding,
                          char* errbuf, size_t errbuf_len);
void qt_tokenizer_free(qt_tokenizer* t);

/* Encode text -> ids. The library allocates *ids; free with qt_ids_free.
 * Returns 0 on success. "_with_special" parses special-token strings. */
int qt_encode(const qt_tokenizer* t, const char* text, size_t len,
              uint32_t** ids, size_t* n);
int qt_encode_with_special(const qt_tokenizer* t, const char* text, size_t len,
                           uint32_t** ids, size_t* n);
void qt_ids_free(uint32_t* ids);

/* Decode ids -> bytes. The library allocates *out (NUL-terminated; *outlen
 * excludes the NUL); free with qt_str_free. Returns 0 on success. */
int qt_decode(const qt_tokenizer* t, const uint32_t* ids, size_t n,
              char** out, size_t* outlen);
void qt_str_free(char* s);

/* There is no batch call in the C ABI: a qt_tokenizer handle is safe to use
 * from many threads concurrently, so batch = your threads + qt_encode. */

/* Token count of text, or (size_t)-1 on error. */
size_t qt_count(const qt_tokenizer* t, const char* text, size_t len);

size_t qt_vocab_size(const qt_tokenizer* t);
const char* qt_encoding(const qt_tokenizer* t);   /* borrowed; valid while t lives */

#ifdef __cplusplus
}
#endif
#endif /* QUICKTOK_H */
