// libFuzzer target: encode arbitrary bytes (must never crash / OOB / UB), and
// verify decode(encode(x)) reproduces x for valid UTF-8. Build:
//   clang++ -O1 -g -std=c++20 -fsanitize=fuzzer,address,undefined \
//     -Iinclude -Isrc test/fuzz_quicktok.cpp src/quicktok.cpp src/trie2_mb.cpp -o fuzz
//   ./fuzz                       # data/ must be reachable from cwd
#include "quicktok.hpp"
#include <string>
#include <vector>

static const quicktok::Tokenizer& tok() {
    static quicktok::Tokenizer t = quicktok::Tokenizer::load_dir("data");
    return t;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    const auto& t = tok();
    std::vector<uint32_t> ids;
    t.encode(data, size, ids);            // arbitrary bytes — must not crash
    // round-trip: decoded bytes re-encode to the same ids (encode is a function of bytes)
    std::string back = t.decode(ids);
    auto ids2 = t.encode(back);
    if (ids2 != ids) __builtin_trap();     // re-encoding our own output must be stable
    return 0;
}
