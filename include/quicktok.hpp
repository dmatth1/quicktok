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
    // "o200k_harmony" (GPT-OSS), "llama3", "qwen3" (Qwen2.5/Qwen3), or "llama4"
    // (bring-your-own gated vocab — see README). dir must contain <stem>.vocab,
    // the matching uniclass file, and (optionally) <stem>.special — see data/ in
    // the repo. Throws std::runtime_error on missing/corrupt files or unknown names.
    static Tokenizer load_dir(const std::string& dir, const std::string& encoding = "cl100k_base");
    // explicit paths (cl100k-pattern encodings only; prefer load_dir)
    static Tokenizer load(const std::string& vocab_path, const std::string& uniclass_path);

    // text -> token ids. Special-token strings in the input are encoded as plain
    // text (tiktoken's encode_ordinary semantics).
    void encode(const uint8_t* text, size_t len, std::vector<uint32_t>& out) const;
    std::vector<uint32_t> encode(std::string_view text) const;

    // like encode(), but occurrences of this encoding's special tokens
    // (e.g. "<|endoftext|>") become their special ids (tiktoken's
    // encode(text, allowed_special="all") semantics).
    std::vector<uint32_t> encode_with_special(std::string_view text) const;

    // number of tokens encode() would produce (same cost as encode)
    size_t count(std::string_view text) const;

    // encode many texts in parallel (ordinary semantics). threads = 0 picks
    // hardware concurrency. Safe because a loaded Tokenizer is shareable.
    std::vector<std::vector<uint32_t>> encode_batch(const std::vector<std::string_view>& texts,
                                                    unsigned threads = 0) const;

    // token ids -> bytes (appends to out). Special ids decode to their strings;
    // other ids >= vocab_size() are skipped.
    void decode(const uint32_t* ids, size_t n, std::string& out) const;
    std::string decode(const std::vector<uint32_t>& ids) const;

    size_t vocab_size() const;          // base vocabulary size (excludes specials)
    const std::string& encoding() const;

    Tokenizer(Tokenizer&&) noexcept;
    Tokenizer& operator=(Tokenizer&&) noexcept;
    ~Tokenizer();

private:
    Tokenizer();
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace quicktok
