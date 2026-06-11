// Optional comparator: tokenize a corpus with LLAMA.CPP's own tokenizer (links libllama, calls
// llama_tokenize) — the reference side of the quicktok-vs-llama.cpp table (see bench/README.md).
// Loads a vocab-only GGUF, times best-of-N, dumps ids to <corpus>.<tag>.ids so
// our file_bench can check exactness on the SAME tokens.
//   clang++ -O3 -std=c++17 llamacpp_bench.cpp -I$LLAMA/include -L$LLAMA/build/bin -lllama -o llamacpp_bench
//   LD_LIBRARY_PATH=$LLAMA/build/bin ./llamacpp_bench <vocab.gguf> <corpus.txt> <tag>
#include "llama.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
using clk = std::chrono::steady_clock;

static std::string rd(const char* p){ FILE* f=fopen(p,"rb"); if(!f){fprintf(stderr,"open %s\n",p);exit(1);} fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET); std::string s; s.resize(n); if(n&&fread(&s[0],1,n,f)!=(size_t)n)exit(1); fclose(f); return s; }
static void silent(enum ggml_log_level, const char*, void*) {}

int main(int argc, char** argv){
    if (argc < 4){ fprintf(stderr,"usage: llamacpp_bench <vocab.gguf> <corpus> <tag>\n"); return 2; }
    llama_log_set(silent, nullptr);
    llama_backend_init();
    auto mp = llama_model_default_params(); mp.vocab_only = true;
    llama_model* m = llama_model_load_from_file(argv[1], mp);
    if (!m){ fprintf(stderr,"load failed\n"); return 1; }
    const llama_vocab* v = llama_model_get_vocab(m);

    std::string text = rd(argv[2]);
    int32_t len = (int32_t)text.size();
    std::vector<int32_t> toks(len + 64);
    auto run = [&]()->int32_t {
        int32_t n = llama_tokenize(v, text.data(), len, toks.data(), (int32_t)toks.size(), false, false);
        if (n < 0) { toks.resize(-n); n = llama_tokenize(v, text.data(), len, toks.data(), (int32_t)toks.size(), false, false); }
        return n;
    };
    int32_t n = run();
    double best = 1e30;
    for (int r=0;r<5;r++){ auto t0=clk::now(); volatile int32_t nn=run(); (void)nn; double s=std::chrono::duration<double>(clk::now()-t0).count(); if(s<best)best=s; }
    double mb = len/1e6;
    printf("LLAMA.CPP tokenizer (%s) on %s: %d tokens (%.3f tok/byte)\n", argv[3], argv[2], n, (double)n/len);
    printf("  llama.cpp     %8.2f MB/s  %6.2f Mtok/s\n", mb/best, n/best/1e6);

    std::string idp = std::string(argv[2]) + "." + argv[3] + ".ids";
    FILE* fo = fopen(idp.c_str(),"wb"); uint32_t un=(uint32_t)n; fwrite(&un,4,1,fo);
    for (int32_t i=0;i<n;i++){ uint32_t id=(uint32_t)toks[i]; fwrite(&id,4,1,fo); } fclose(fo);
    llama_model_free(m);
    return 0;
}
