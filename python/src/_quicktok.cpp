// Python binding (pybind11) over the quicktok C++ API. Data files are packaged
// inside the wheel (quicktok/data/); the module resolves them at import.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cstring>
#include "quicktok.hpp"
#include <string>
#include <set>
#include <algorithm>

namespace py = pybind11;

static std::string g_datadir;   // set by the package on import

class PyTokenizer {
public:
    PyTokenizer(const std::string& encoding, const std::string& datadir)
        : tok(quicktok::Tokenizer::load_dir(datadir.empty() ? g_datadir : datadir, encoding)) {}

    std::vector<uint32_t> encode(py::bytes b) const {
        char* d; Py_ssize_t n; PYBIND11_BYTES_AS_STRING_AND_SIZE(b.ptr(), &d, &n);
        py::gil_scoped_release rel;
        return tok.encode(std::string_view(d, (size_t)n));
    }
    std::vector<uint32_t> encode_str(const std::string& s) const {
        py::gil_scoped_release rel;
        return tok.encode(std::string_view(s));
    }
    std::vector<uint32_t> encode_with_special(const std::string& s) const {
        py::gil_scoped_release rel;
        return tok.encode_with_special(std::string_view(s));
    }
    // ordinary encode + per-token spans. unit="byte" (exact UTF-8 byte offsets,
    // tile the input) or "char" (code-point offsets, HF offset_mapping shape).
    py::tuple encode_with_offsets(const std::string& text, const std::string& unit) const {
        if (unit != "byte" && unit != "char")
            throw py::value_error("unit must be 'byte' or 'char'");
        std::vector<uint32_t> ids, bounds;
        {
            py::gil_scoped_release rel;
            tok.encode_with_offsets((const uint8_t*)text.data(), text.size(), ids, bounds);
        }
        py::list spans;
        if (unit == "byte") {
            for (size_t i = 0; i + 1 < bounds.size(); i++)
                spans.append(py::make_tuple(bounds[i], bounds[i + 1]));
        } else {
            // code-point index of each byte position (count of UTF-8 lead bytes).
            // Exact for non-NFC encodings, where the encoded bytes == input bytes.
            size_t n = text.size();
            std::vector<uint32_t> b2c(n + 1);
            uint32_t c = 0;
            for (size_t i = 0; i < n; i++) { b2c[i] = c; if (((uint8_t)text[i] & 0xC0) != 0x80) c++; }
            b2c[n] = c;
            for (size_t i = 0; i + 1 < bounds.size(); i++)
                spans.append(py::make_tuple(bounds[i] <= n ? b2c[bounds[i]] : c,
                                            bounds[i + 1] <= n ? b2c[bounds[i + 1]] : c));
        }
        return py::make_tuple(ids, spans);
    }
    // tiktoken's Encoding.encode(text, *, allowed_special=set(), disallowed_special="all"):
    // raise if a disallowed special string occurs; otherwise encode, turning the
    // allowed specials into their ids and everything else into ordinary tokens.
    std::vector<uint32_t> encode_tt(const std::string& text, const py::object& allowed,
                                    const py::object& disallowed) const {
        const auto& sp = tok.special_tokens();
        auto is_all = [](const py::object& o){ return py::isinstance<py::str>(o) && o.cast<std::string>() == "all"; };
        auto to_set = [](const py::object& o){
            std::set<std::string> r;
            for (py::handle h : o) r.insert(h.cast<std::string>());
            return r;
        };
        std::set<std::string> allow;
        if (is_all(allowed)) { for (const auto& pr : sp) allow.insert(pr.first); }
        else if (!allowed.is_none()) allow = to_set(allowed);
        std::set<std::string> disallow;
        if (is_all(disallowed)) { for (const auto& pr : sp) if (!allow.count(pr.first)) disallow.insert(pr.first); }
        else if (!disallowed.is_none()) disallow = to_set(disallowed);
        // disallowed check (tiktoken raises ValueError naming the offending token)
        for (const auto& s : disallow) {
            if (s.empty()) continue;
            if (text.find(s) != std::string::npos) {
                std::string q = py::repr(py::str(s)).cast<std::string>();
                throw py::value_error(
                    "Encountered text corresponding to disallowed special token " + q + ".\n"
                    "If you want this text to be encoded as a special token, pass it to "
                    "`allowed_special`, e.g. `allowed_special={" + q + ", ...}`.\n"
                    "If you want this text to be encoded as normal text, disable the check "
                    "for this token by passing `disallowed_special=()`, or use `encode_ordinary`.");
            }
        }
        py::gil_scoped_release rel;
        std::vector<uint32_t> out; out.reserve(text.size() / 3 + 8);
        size_t p = 0;
        while (p < text.size()) {
            size_t best = std::string::npos, bi = 0;
            for (size_t i = 0; i < sp.size(); i++)
                if (allow.count(sp[i].first)) {
                    size_t q = text.find(sp[i].first, p);
                    if (q < best) { best = q; bi = i; }
                }
            if (best == std::string::npos) {
                tok.encode((const uint8_t*)text.data() + p, text.size() - p, out); break;
            }
            if (best > p) tok.encode((const uint8_t*)text.data() + p, best - p, out);
            out.push_back(sp[bi].second);
            p = best + sp[bi].first.size();
        }
        return out;
    }
    // Same as encode(), but returns a numpy uint32 array instead of a list[int].
    // For large inputs this is much faster: the result is one contiguous buffer
    // (a single memcpy) rather than millions of per-token Python int objects — the
    // list-building cost that otherwise dominates the wheel's time on big inputs.
    // tiktoken's encode_to_numpy.
    py::array_t<uint32_t> encode_to_numpy(const std::string& text, const py::object& allowed,
                                          const py::object& disallowed) const {
        std::vector<uint32_t> ids = encode_tt(text, allowed, disallowed);   // GIL released inside
        py::array_t<uint32_t> arr((py::ssize_t)ids.size());
        if (!ids.empty()) std::memcpy(arr.mutable_data(), ids.data(), ids.size() * sizeof(uint32_t));
        return arr;
    }
    // exact bytes/str of a SINGLE token -> its id (tiktoken's encode_single_token).
    uint32_t encode_single_token(const py::object& piece) const {
        std::string b;
        if (py::isinstance<py::bytes>(piece)) {
            char* d; Py_ssize_t n; PYBIND11_BYTES_AS_STRING_AND_SIZE(piece.ptr(), &d, &n);
            b.assign(d, (size_t)n);
        } else {
            b = piece.cast<std::string>();   // str -> utf-8
        }
        int64_t id = tok.token_id(b);
        if (id < 0) throw py::key_error("bytes are not a single token in this encoding");
        return (uint32_t)id;
    }
    bool is_special_token(uint32_t id) const {
        for (const auto& [s, sid] : tok.special_tokens()) if (sid == id) return true;
        return false;
    }
    uint32_t max_token_value() const { return (uint32_t)(tok.n_vocab() - 1); }
    py::list token_byte_values() const {   // base-vocab token bytes, lexicographically
        std::vector<std::string> vals(tok.vocab_size());   // tiktoken returns sorted_token_bytes
        for (uint32_t i = 0; i < (uint32_t)tok.vocab_size(); i++) tok.decode(&i, 1, vals[i]);
        std::sort(vals.begin(), vals.end());
        py::list r;
        for (auto& v : vals) r.append(py::bytes(v));
        return r;
    }
    py::list decode_batch(const std::vector<std::vector<uint32_t>>& batch, const std::string& errors) const {
        py::list r;
        for (const auto& ids : batch) r.append(decode(ids, errors));
        return r;
    }
    size_t count(const std::string& s) const {
        py::gil_scoped_release rel;
        return tok.count(std::string_view(s));
    }
    void check_ids(const std::vector<uint32_t>& ids) const {
        for (uint32_t id : ids)
            if (!tok.known_id(id))
                throw py::key_error("token id " + std::to_string(id) + " is not in the vocabulary");
    }
    py::bytes decode_bytes(const std::vector<uint32_t>& ids) const {
        check_ids(ids);
        std::string out;
        { py::gil_scoped_release rel; tok.decode(ids.data(), ids.size(), out); }
        return py::bytes(out);
    }
    py::str decode(const std::vector<uint32_t>& ids, const std::string& errors) const {
        check_ids(ids);
        std::string out;
        { py::gil_scoped_release rel; tok.decode(ids.data(), ids.size(), out); }
        // tiktoken parity: decode() defaults to errors="replace" (lossy on tokens
        // that split a UTF-8 character); use decode_bytes() for the exact bytes.
        PyObject* s = PyUnicode_DecodeUTF8(out.data(), (Py_ssize_t)out.size(), errors.c_str());
        if (!s) throw py::error_already_set();
        return py::reinterpret_steal<py::str>(s);
    }
    py::bytes decode_single_token_bytes(uint32_t id) const {
        check_ids({id});
        std::string out;
        tok.decode(&id, 1, out);
        return py::bytes(out);
    }
    uint32_t eot_token() const {
        for (const char* name : {"<|endoftext|>", "<|end_of_text|>", "</s>"})
            for (const auto& [s, sid] : tok.special_tokens())
                if (s == name) return sid;
        throw py::key_error("this encoding has no end-of-text special token");
    }
    py::set special_tokens_set() const {
        py::set out;
        for (const auto& [s, sid] : tok.special_tokens()) out.add(py::str(s));
        return out;
    }
    // flat batch: encode in parallel (GIL released), return one contiguous uint32
    // token buffer + an int64 offsets array (len n+1) instead of n Python lists.
    // doc i's tokens = tokens[offsets[i]:offsets[i+1]]. Avoids per-document Python
    // object construction — the marshalling cost that would cap a list-returning path.
    py::tuple encode_batch(const std::vector<std::string>& texts, unsigned threads,
                           bool with_special) const {
        std::vector<std::vector<uint32_t>> res;
        {
            std::vector<std::string_view> views(texts.begin(), texts.end());
            py::gil_scoped_release rel;
            res = tok.encode_batch(views, threads, with_special);
        }
        size_t n = res.size(), total = 0;
        for (const auto& v : res) total += v.size();
        py::array_t<uint32_t> tokens((py::ssize_t)total);
        py::array_t<int64_t> offsets((py::ssize_t)(n + 1));
        uint32_t* tp = tokens.mutable_data();
        int64_t* op = offsets.mutable_data();
        size_t pos = 0; op[0] = 0;
        for (size_t i = 0; i < n; i++) {
            if (!res[i].empty()) std::memcpy(tp + pos, res[i].data(), res[i].size() * sizeof(uint32_t));
            pos += res[i].size();
            op[i + 1] = (int64_t)pos;
        }
        return py::make_tuple(std::move(tokens), std::move(offsets));
    }
    size_t vocab_size() const { return tok.vocab_size(); }
    size_t n_vocab() const { return tok.n_vocab(); }
    std::string name() const { return tok.encoding(); }

private:
    quicktok::Tokenizer tok;
};

PYBIND11_MODULE(_quicktok, m) {
    m.attr("__version__") = "0.4.0";
    m.def("_set_datadir", [](const std::string& d){ g_datadir = d; });

    py::class_<PyTokenizer>(m, "Tokenizer")
        .def(py::init<const std::string&, const std::string&>(),
             py::arg("encoding"), py::arg("data_dir") = "")
        .def("encode", &PyTokenizer::encode_tt, py::arg("text"),
             py::kw_only(), py::arg("allowed_special") = py::set(),
             py::arg("disallowed_special") = py::str("all"),
             "Encode text -> token ids (tiktoken semantics). By default a special-token "
             "string in the input raises ValueError; pass allowed_special=\"all\" (or a set) "
             "to encode specials as their ids, or use encode_ordinary() to treat them as text. "
             "disallowed_special=() disables the check.")
        .def("encode_ordinary", &PyTokenizer::encode_str, py::arg("text"),
             "Encode text -> token ids, treating special-token strings as plain text "
             "(never raises). tiktoken's encode_ordinary.")
        .def("encode_to_numpy", &PyTokenizer::encode_to_numpy, py::arg("text"),
             py::kw_only(), py::arg("allowed_special") = py::set(),
             py::arg("disallowed_special") = py::str("all"),
             "Like encode(), but returns a numpy uint32 array (one contiguous buffer, "
             "no per-token Python objects) — much faster on large inputs. tiktoken's "
             "encode_to_numpy. For ordinary-text throughput pass disallowed_special=().")
        .def("encode_with_special", &PyTokenizer::encode_with_special, py::arg("text"),
             "Encode, turning every known special-token string into its id "
             "(== encode(text, allowed_special=\"all\")).")
        .def("encode_with_special_tokens", &PyTokenizer::encode_with_special, py::arg("text"),
             "Alias of encode_with_special() — the name tiktoken-rs / TokenDagger use.")
        .def("encode_with_offsets", &PyTokenizer::encode_with_offsets, py::arg("text"),
             py::kw_only(), py::arg("unit") = "byte",
             "Ordinary encode -> (ids, spans). spans[i] = (start, end) for token i. "
             "unit='byte' (default): exact UTF-8 byte offsets that tile the input. "
             "unit='char': code-point offsets (HuggingFace offset_mapping shape; "
             "exact for non-NFC encodings).")
        .def("encode_single_token", &PyTokenizer::encode_single_token, py::arg("text_or_bytes"),
             "Id of a piece that is exactly one token (str or bytes); KeyError otherwise.")
        .def("is_special_token", &PyTokenizer::is_special_token, py::arg("id"),
             "True iff the id is one of this encoding's special tokens.")
        .def("decode_batch", &PyTokenizer::decode_batch, py::arg("batch"), py::arg("errors") = "replace",
             "Decode many id lists -> list[str].")
        .def("token_byte_values", &PyTokenizer::token_byte_values,
             "List of the base vocabulary's token byte-strings, indexed by rank.")
        .def("encode_batch", &PyTokenizer::encode_batch, py::arg("texts"), py::arg("threads") = 0,
             py::arg("with_special") = false,
             "Encode many texts in parallel -> (tokens uint32[], offsets int64[]); "
             "doc i is tokens[offsets[i]:offsets[i+1]]. with_special=True parses "
             "special-token strings (chat-templated data).")
        .def("count", &PyTokenizer::count, py::arg("text"))
        .def("decode", &PyTokenizer::decode, py::arg("ids"), py::arg("errors") = "replace",
             "Decode ids -> str. Unknown ids raise KeyError; tokens that split a "
             "UTF-8 character decode per `errors` (default 'replace', like tiktoken).")
        .def("decode_bytes", &PyTokenizer::decode_bytes, py::arg("ids"),
             "Decode ids -> exact bytes (lossless). Unknown ids raise KeyError.")
        .def("decode_single_token_bytes", &PyTokenizer::decode_single_token_bytes, py::arg("id"),
             "Bytes of one token id (KeyError if unknown) — matches tiktoken's name.")
        .def_property_readonly("eot_token", &PyTokenizer::eot_token,
             "Id of the end-of-text special token (KeyError if the encoding has none).")
        .def_property_readonly("special_tokens_set", &PyTokenizer::special_tokens_set,
             "The special-token strings of this encoding, as a set.")
        .def_property_readonly("name", &PyTokenizer::name)
        .def_property_readonly("n_vocab", &PyTokenizer::n_vocab,
             "Max token id + 1, specials included (tiktoken semantics; cl100k -> 100277).")
        .def_property_readonly("max_token_value", &PyTokenizer::max_token_value,
             "Highest token id, specials included (== n_vocab - 1); tiktoken's name.")
        .def("__repr__", [](const PyTokenizer& t){ return "<quicktok.Tokenizer '" + t.name() + "'>"; });
}
