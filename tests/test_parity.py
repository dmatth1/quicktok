"""Byte-exact parity of the Python API vs tiktoken, across the full surface.
The 0.4.0 release is defined by this surface (encode() raise semantics +
~10 parity methods), so it gets the most coverage."""
import pytest
import quicktok
from conftest import TIKTOKEN_ENCODINGS, SAMPLES


@pytest.fixture(params=TIKTOKEN_ENCODINGS)
def pair(request, tiktoken_mod):
    name = request.param
    return name, quicktok.get_encoding(name), tiktoken_mod.get_encoding(name)


@pytest.mark.parametrize("s", SAMPLES)
def test_encode_ordinary_matches_tiktoken(pair, s):
    name, q, t = pair
    assert q.encode_ordinary(s) == t.encode_ordinary(s), name


@pytest.mark.parametrize("s", SAMPLES)
def test_decode_roundtrip(pair, s):
    _, q, _ = pair
    assert q.decode(q.encode_ordinary(s)) == s
    # decode_bytes is the lossless path
    assert q.decode_bytes(q.encode_ordinary(s)) == s.encode("utf-8")


def test_encode_raises_on_special_by_default(pair):
    # the headline 0.4.0 behavior change — matches tiktoken's default guard
    name, q, t = pair
    s = "before <|endoftext|> after"
    with pytest.raises(ValueError):
        q.encode(s)
    with pytest.raises(ValueError):
        t.encode(s)  # tiktoken does the same


def test_encode_allowed_special_all_matches_tiktoken(pair):
    name, q, t = pair
    s = "a <|endoftext|> b"
    assert q.encode(s, allowed_special="all") == t.encode(s, allowed_special="all"), name


def test_encode_disallowed_empty_equals_ordinary(pair):
    _, q, _ = pair
    s = "x <|endoftext|> y"
    assert q.encode(s, disallowed_special=()) == q.encode_ordinary(s)


def test_encode_with_special_tokens_alias(pair):
    name, q, t = pair
    s = "a <|endoftext|> b"
    ref = t.encode(s, allowed_special="all")
    assert q.encode_with_special(s) == ref
    assert q.encode_with_special_tokens(s) == ref  # tiktoken-rs/TokenDagger name


def test_vocab_invariants_match_tiktoken(pair):
    name, q, t = pair
    assert q.n_vocab == t.n_vocab, name
    assert q.max_token_value == t.max_token_value, name
    assert q.eot_token == t.eot_token, name
    assert set(q.special_tokens_set) == set(t.special_tokens_set), name


def test_single_token_helpers_match_tiktoken(pair):
    name, q, t = pair
    for piece in (b"hello", b" the", b"123"):
        assert q.encode_single_token(piece) == t.encode_single_token(piece), (name, piece)
    tid = q.encode_single_token(b"hello")
    assert q.decode_single_token_bytes(tid) == t.decode_single_token_bytes(tid)


def test_token_byte_values_match_tiktoken(pair):
    name, q, t = pair
    assert q.token_byte_values() == t.token_byte_values(), name


def test_is_special_token(pair):
    _, q, _ = pair
    assert q.is_special_token(q.eot_token)
    assert not q.is_special_token(q.encode_single_token(b"hello"))


def test_decode_batch_matches_tiktoken(pair):
    name, q, t = pair
    batch = [q.encode_ordinary(s) for s in SAMPLES if s]
    assert q.decode_batch(batch) == t.decode_batch(batch), name


def test_decode_unknown_id_raises(pair):
    _, q, _ = pair
    with pytest.raises((KeyError, ValueError)):
        q.decode([q.max_token_value + 999])
