# C++ API

[← back to README](../README.md)

## Build

```sh
git clone https://github.com/dmatth1/quicktok
cd quicktok
make            # builds build/libquicktok.{a, dylib/so}
make test       # verifies exact ids vs the references (all encodings) + C ABI
```

Or consume it from another CMake project via `find_package` or `FetchContent`,
or `make install` + pkg-config:

```cmake
find_package(quicktok REQUIRED)
target_link_libraries(app PRIVATE quicktok::quicktok)
```

## Usage

```cpp
#include <quicktok.hpp>

auto tok = quicktok::Tokenizer::load_dir("data");              // cl100k_base
auto gpt4o = quicktok::Tokenizer::load_dir("data", "o200k_base");

auto ids = tok.encode("Hello, quicktok! 日本語 🚀");  // std::vector<uint32_t>
std::string text = tok.decode(ids);                   // lossless round-trip
size_t n = tok.count("how many tokens is this?");
auto with_sp = tok.encode_with_special("a<|endoftext|>b");  // specials -> ids
auto batch = tok.encode_batch(texts);                       // parallel
```

## `class Tokenizer`

```cpp
namespace quicktok {
class Tokenizer {
    // encoding: a built-in name (see Encodings) or any imported one
    static Tokenizer load_dir(const std::string& dir, const std::string& encoding = "cl100k_base");

    std::vector<uint32_t> encode(std::string_view text) const;          // encode_ordinary semantics
    std::vector<uint32_t> encode_with_special(std::string_view) const;  // allowed_special="all"
    void encode(const uint8_t* text, size_t len, std::vector<uint32_t>& out) const;
    std::vector<std::vector<uint32_t>> encode_batch(const std::vector<std::string_view>&,
                                                    unsigned threads = 0, bool with_special = false) const;
    size_t count(std::string_view text) const;

    std::string decode(const std::vector<uint32_t>& ids) const;         // handles special ids too
    void decode(const uint32_t* ids, size_t n, std::string& out) const;

    size_t vocab_size() const;     // base vocab; n_vocab() = max id + 1 incl. specials
    size_t n_vocab() const;
    const std::vector<std::pair<std::string, uint32_t>>& special_tokens() const;
    const std::string& encoding() const;
};
}
```

- `load_dir` takes the directory holding the data files (`data/` in this repo;
  `make install` puts them in `share/quicktok`). It throws `std::runtime_error`
  on missing or corrupt files; nothing throws on the encode hot path (one
  exception: inputs over 4 GiB per call are rejected).
- A loaded `Tokenizer` is safe to share across threads — concurrent
  `encode()`/`decode()` is supported and tested.

## C ABI

There's also a stable **C ABI** (`include/quicktok.h`) for FFI from any language.
It mirrors the C++ surface (load / encode / decode / count / free) with C linkage
and out-parameters instead of return values; see the header for the full
declaration set.

## Encodings

Built-in names (`cl100k_base`, `o200k_base`, `o200k_harmony`, `llama3`, `qwen3`)
and importing other tokenizers are covered in [docs/encodings.md](encodings.md).
