/* C ABI smoke test — compiled as C, links the C++ static lib. */
#include "quicktok.h"
#include <stdio.h>
#include <string.h>

int main(void) {
    char err[256] = {0};
    qt_tokenizer* t = qt_load_dir("data", "cl100k_base", err, sizeof err);
    if (!t) { printf("FAIL load: %s\n", err); return 1; }

    const char* text = "Hello from C! 123";
    uint32_t* ids; size_t n;
    if (qt_encode(t, text, strlen(text), &ids, &n) || n == 0) { printf("FAIL encode\n"); return 1; }

    char* back; size_t blen;
    if (qt_decode(t, ids, n, &back, &blen) || strcmp(back, text) != 0) { printf("FAIL decode\n"); return 1; }

    if (qt_count(t, text, strlen(text)) != n) { printf("FAIL count\n"); return 1; }
    if (qt_vocab_size(t) != 100256) { printf("FAIL vocab_size\n"); return 1; }

    qt_tokenizer* bad = qt_load_dir("/nope", "cl100k_base", err, sizeof err);
    if (bad || err[0] == 0) { printf("FAIL error path\n"); return 1; }

    qt_str_free(back); qt_ids_free(ids); qt_tokenizer_free(t);
    printf("C ABI: OK (%zu tokens, round-trip, errors reported)\n", n);
    return 0;
}
