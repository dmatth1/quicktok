// quicktok — fast exact BPE tokenizer for OpenAI encodings.
// Token ids are byte-exact vs tiktoken (the reference IS the spec).
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace quicktok {

class Tokenizer {
public:
    // encoding: "cl100k_base" (GPT-3.5/4), "o200k_base" (GPT-4o),
    // "o200k_harmony" (GPT-OSS), "llama3", "qwen3" (Qwen2.5/Qwen3), "llama4"
    // (bring-your-own gated vocab), or any name imported by
    // tools/import_tokenizer.py (loaded via its <name>.enc sidecar). dir must
    // contain <stem>.vocab, the matching uniclass/nfc tables, and (optionally)
    // <stem>.special — see data/ in the repo. Throws std::runtime_error on
    // missing/corrupt files or unknown names.
    static Tokenizer load_dir(const std::string& dir, const std::string& encoding = "cl100k_base");
    // explicit paths (cl100k-pattern encodings only; prefer load_dir)
    static Tokenizer load(const std::string& vocab_path, const std::string& uniclass_path);

    // text -> token ids. Special-token strings in the input are encoded as plain
    // text (tiktoken's encode_ordinary semantics). Never raises on specials —
    // the raise-on-disallowed behavior of tiktoken's Python encode() lives in the
    // Python binding (this C++ surface mirrors tiktoken-rs, which also doesn't raise).
    void encode(const uint8_t* text, size_t len, std::vector<uint32_t>& out) const;
    std::vector<uint32_t> encode(std::string_view text) const;
    // explicit name for the ordinary path (tiktoken / tiktoken-rs / TokenDagger
    // all spell it `encode_ordinary`); identical to encode(std::string_view).
    std::vector<uint32_t> encode_ordinary(std::string_view text) const { return encode(text); }

    // like encode(), but occurrences of this encoding's special tokens
    // (e.g. "<|endoftext|>") become their special ids (tiktoken's
    // encode(text, allowed_special="all") semantics).
    std::vector<uint32_t> encode_with_special(std::string_view text) const;
    // alias under the name tiktoken-rs / TokenDagger use.
    std::vector<uint32_t> encode_with_special_tokens(std::string_view text) const { return encode_with_special(text); }

    // number of tokens encode() would produce (same cost as encode)
    size_t count(std::string_view text) const;

    // encode many texts in parallel. threads = 0 picks hardware concurrency;
    // with_special = true gives encode_with_special() semantics per text (e.g.
    // chat-templated fine-tuning data). Safe because a loaded Tokenizer is shareable.
    std::vector<std::vector<uint32_t>> encode_batch(const std::vector<std::string_view>& texts,
                                                    unsigned threads = 0,
                                                    bool with_special = false) const;

    // token ids -> bytes (appends to out). Special ids decode to their strings;
    // other ids >= vocab_size() are skipped.
    void decode(const uint32_t* ids, size_t n, std::string& out) const;
    std::string decode(const std::vector<uint32_t>& ids) const;

    size_t vocab_size() const;          // base vocabulary size (excludes specials)
    size_t n_vocab() const;             // tiktoken's n_vocab: max token id + 1, INCLUDING specials
    const std::string& encoding() const;

    // this encoding's special tokens (string, id), sorted by id
    const std::vector<std::pair<std::string, uint32_t>>& special_tokens() const;
    // true iff id decodes to something (a vocab token or a special)
    bool known_id(uint32_t id) const;
    // exact bytes -> model id, or -1 if those bytes are not a single token (base
    // vocab or a special). Backs the Python encode_single_token().
    int64_t token_id(std::string_view token_bytes) const;

    Tokenizer(Tokenizer&&) noexcept;
    Tokenizer& operator=(Tokenizer&&) noexcept;
    ~Tokenizer();

private:
    Tokenizer();
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace quicktok
