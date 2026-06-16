#include <quicktok.hpp>   // resolved via find_package include dirs
#include <cstdio>
// argv[1] = installed data dir (share/quicktok)
int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: consumer <data_dir>\n"); return 2; }
    auto tok = quicktok::Tokenizer::load_dir(argv[1]);            // cl100k_base
    auto ids = tok.encode_ordinary("find_package(quicktok) works 123");
    if (tok.decode(ids) != "find_package(quicktok) works 123") { fprintf(stderr, "roundtrip failed\n"); return 1; }
    printf("consumer OK: %zu tokens, vocab %zu\n", ids.size(), tok.vocab_size());
    return 0;
}
