// Python binding (pybind11) over the quicktok C++ API. Data files are packaged
// inside the wheel (quicktok/data/); the module resolves them at import.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "quicktok.hpp"
#include <string>

namespace py = pybind11;

static std::string g_datadir;   // set by the package on import

class PyTokenizer {
public:
    explicit PyTokenizer(const std::string& encoding)
        : tok(quicktok::Tokenizer::load_dir(g_datadir, encoding)) {}

    std::vector<uint32_t> encode(py::bytes b) const {
        char* d; ssize_t n; PYBIND11_BYTES_AS_STRING_AND_SIZE(b.ptr(), &d, &n);
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
    py::bytes decode_bytes(const std::vector<uint32_t>& ids) const {
        std::string out;
        { py::gil_scoped_release rel; tok.decode(ids.data(), ids.size(), out); }
        return py::bytes(out);
    }
    std::string decode(const std::vector<uint32_t>& ids) const {
        std::string out;
        { py::gil_scoped_release rel; tok.decode(ids.data(), ids.size(), out); }
        return out;   // utf-8 -> str
    }
    std::vector<std::vector<uint32_t>> encode_batch(const std::vector<std::string>& texts,
                                                    unsigned threads) const {
        std::vector<std::string_view> views(texts.begin(), texts.end());
        py::gil_scoped_release rel;
        return tok.encode_batch(views, threads);
    }
    size_t vocab_size() const { return tok.vocab_size(); }
    std::string name() const { return tok.encoding(); }

private:
    quicktok::Tokenizer tok;
};

PYBIND11_MODULE(_quicktok, m) {
    m.attr("__version__") = "0.3.0";
    m.def("_set_datadir", [](const std::string& d){ g_datadir = d; });

    py::class_<PyTokenizer>(m, "Tokenizer")
        .def(py::init<const std::string&>(), py::arg("encoding"))
        .def("encode", &PyTokenizer::encode_str, py::arg("text"),
             "Encode text -> token ids (ordinary semantics; specials are plain text).")
        .def("encode", &PyTokenizer::encode, py::arg("text"))
        .def("encode_ordinary", &PyTokenizer::encode_str, py::arg("text"),
             "Alias of encode() — matches tiktoken's name.")
        .def("encode_with_special", &PyTokenizer::encode_with_special, py::arg("text"),
             "Encode, turning known special-token strings into their ids.")
        .def("encode_batch", &PyTokenizer::encode_batch, py::arg("texts"), py::arg("threads") = 0,
             "Encode many texts in parallel.")
        .def("count", &PyTokenizer::count, py::arg("text"))
        .def("decode", &PyTokenizer::decode, py::arg("ids"),
             "Decode ids -> str (utf-8).")
        .def("decode_bytes", &PyTokenizer::decode_bytes, py::arg("ids"),
             "Decode ids -> bytes (never fails on partial multibyte sequences).")
        .def_property_readonly("name", &PyTokenizer::name)
        .def_property_readonly("n_vocab", &PyTokenizer::vocab_size)
        .def("__repr__", [](const PyTokenizer& t){ return "<quicktok.Tokenizer '" + t.name() + "'>"; });
}
