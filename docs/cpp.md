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
    // explicit paths (cl100k-pattern encodings only; prefer load_dir)
    static Tokenizer load(const std::string& vocab_path, const std::string& uniclass_path);

    // encode_ordinary semantics — specials in the input are encoded as plain text;
    // never raises on specials (this C++ surface mirrors tiktoken-rs).
    std::vector<uint32_t> encode(std::string_view text) const;
    void encode(const uint8_t* text, size_t len, std::vector<uint32_t>& out) const;
    std::vector<uint32_t> encode_ordinary(std::string_view text) const;       // explicit alias for encode()

    // specials in the input become their ids (allowed_special="all" semantics)
    std::vector<uint32_t> encode_with_special(std::string_view text) const;
    std::vector<uint32_t> encode_with_special_tokens(std::string_view text) const;  // tiktoken-rs/TokenDagger name

    // encode many texts in parallel; threads = 0 picks hardware concurrency
    std::vector<std::vector<uint32_t>> encode_batch(const std::vector<std::string_view>&,
                                                    unsigned threads = 0, bool with_special = false) const;
    size_t count(std::string_view text) const;                  // tokens encode() would produce

    std::string decode(const std::vector<uint32_t>& ids) const; // special ids decode to their strings
    void decode(const uint32_t* ids, size_t n, std::string& out) const;

    size_t vocab_size() const;     // base vocab (excludes specials)
    size_t n_vocab() const;        // tiktoken's n_vocab: max id + 1, INCLUDING specials
    const std::string& encoding() const;
    const std::vector<std::pair<std::string, uint32_t>>& special_tokens() const;  // (string, id), sorted by id
    bool known_id(uint32_t id) const;              // true iff id decodes (vocab token or special)
    int64_t token_id(std::string_view bytes) const;  // exact bytes -> id, or -1 if not a single token
};
}
```

- `load_dir` takes the directory holding the data files (`data/` in this repo;
  `make install` puts them in `share/quicktok`). It throws `std::runtime_error`
  on missing or corrupt files or an unknown encoding name; nothing throws on the
  encode hot path (one exception: inputs over 4 GiB per call are rejected).
- A loaded `Tokenizer` is safe to share across threads — concurrent
  `encode()`/`decode()` is supported and tested.

## C ABI

A stable **C ABI** (`include/quicktok.h`) for FFI from any language — C linkage,
out-parameters instead of return values, library-allocated buffers freed with the
matching `*_free`:

```c
qt_tokenizer* qt_load_dir(const char* dir, const char* encoding, char* errbuf, size_t errbuf_len);
void          qt_tokenizer_free(qt_tokenizer* t);

int  qt_encode(const qt_tokenizer*, const char* text, size_t len, uint32_t** ids, size_t* n);
int  qt_encode_with_special(const qt_tokenizer*, const char* text, size_t len, uint32_t** ids, size_t* n);
void qt_ids_free(uint32_t* ids);

int  qt_decode(const qt_tokenizer*, const uint32_t* ids, size_t n, char** out, size_t* outlen);
void qt_str_free(char* s);

size_t      qt_count(const qt_tokenizer*, const char* text, size_t len);
size_t      qt_vocab_size(const qt_tokenizer*);
const char* qt_encoding(const qt_tokenizer*);
```

Constructors return `NULL` and write a message into `errbuf` on error; other calls
return nonzero / `(size_t)-1`. There is deliberately **no batch call** — a handle is
safe to use from many threads concurrently, so batch = your threads + `qt_encode`.

## Encodings

Built-in names (`cl100k_base`, `o200k_base`, `o200k_harmony`, `llama3`, `qwen3`)
and importing other tokenizers are covered in [docs/encodings.md](encodings.md).
