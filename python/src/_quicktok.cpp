// Python binding (pybind11) over the quicktok C++ API. Data files are packaged
// inside the wheel (quicktok/data/); the module resolves them at import.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cstring>
#include "quicktok.hpp"
#include <string>

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
    m.attr("__version__") = "0.3.1";
    m.def("_set_datadir", [](const std::string& d){ g_datadir = d; });

    py::class_<PyTokenizer>(m, "Tokenizer")
        .def(py::init<const std::string&, const std::string&>(),
             py::arg("encoding"), py::arg("data_dir") = "")
        .def("encode", &PyTokenizer::encode_str, py::arg("text"),
             "Encode text -> token ids (ordinary semantics; specials are plain text).")
        .def("encode", &PyTokenizer::encode, py::arg("text"))
        .def("encode_ordinary", &PyTokenizer::encode_str, py::arg("text"),
             "Alias of encode() — matches tiktoken's name.")
        .def("encode_with_special", &PyTokenizer::encode_with_special, py::arg("text"),
             "Encode, turning known special-token strings into their ids.")
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
        .def("__repr__", [](const PyTokenizer& t){ return "<quicktok.Tokenizer '" + t.name() + "'>"; });
}
