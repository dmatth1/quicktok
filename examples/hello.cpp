#include "quicktok.hpp"
#include <cstdio>

int main() {
    // data/ ships with the repo: cl100k.vocab + uniclass.bin
    auto tok = quicktok::Tokenizer::load_dir("data");
    const char* text = "Hello, quicktok! 日本語 🚀";

    auto ids = tok.encode(text);
    printf("%s\n  -> %zu tokens:", text, ids.size());
    for (auto id : ids) printf(" %u", id);
    printf("\n  -> decoded: %s\n", tok.decode(ids).c_str());
    return 0;
}
