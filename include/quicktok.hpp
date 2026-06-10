// quicktok — fast exact cl100k (GPT-3.5/4) BPE tokenizer.
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
    // dir must contain cl100k.vocab and uniclass.bin (see data/ in the repo).
    static Tokenizer load_dir(const std::string& dir);
    static Tokenizer load(const std::string& vocab_path, const std::string& uniclass_path);

    // text -> token ids (appends to out)
    void encode(const uint8_t* text, size_t len, std::vector<uint32_t>& out) const;
    std::vector<uint32_t> encode(std::string_view text) const;

    // token ids -> bytes (appends to out). Ids >= vocab_size() are skipped.
    void decode(const uint32_t* ids, size_t n, std::string& out) const;
    std::string decode(const std::vector<uint32_t>& ids) const;

    size_t vocab_size() const;

    Tokenizer(Tokenizer&&) noexcept;
    Tokenizer& operator=(Tokenizer&&) noexcept;
    ~Tokenizer();

private:
    Tokenizer();
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace quicktok
