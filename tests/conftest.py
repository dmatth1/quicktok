import pytest

# Encodings that have a direct tiktoken reference (for byte-exact parity tests).
TIKTOKEN_ENCODINGS = ["cl100k_base", "o200k_base"]
# All encodings bundled in the wheel (parity-tested where a reference exists;
# otherwise round-trip / invariant tested).
BUNDLED = ["cl100k_base", "o200k_base", "o200k_harmony", "llama3", "qwen3"]

SAMPLES = [
    "Hello, world! 123",
    "The quick brown fox jumps over the lazy dog.",
    "def f(x): return x*2  # 1234567890",
    "café résumé naïve — Zürich",
    "日本語のトークナイザー 中文 mixed",
    "emoji 🚀🔥👍 and  spaces\n\ttabs",
    "don't can't I'll we've",
    "https://example.com/p?q=1&x=2",
    "",
    "a",
    "\n\n\n",
    "ALLCAPS lower MixedCase 42 3.14",
]


@pytest.fixture(scope="session")
def tiktoken_mod():
    return pytest.importorskip("tiktoken")
