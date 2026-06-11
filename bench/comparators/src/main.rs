// bpe-openai + tiktoken-rs on one corpus file — the Rust comparator rows of the
// README tables. Single thread, best-of-5, each called through the same raw API
// its own benchmark uses (bpe's Tokenizer::encode, tiktoken-rs encode_ordinary).
// Both are exact-checked against a reference ids file (u32 LE count + ids,
// written by bench/compare.py from tiktoken) before timing.
//
//   cargo run --release -- <corpus.txt> <cl100k|o200k> <ref.ids>
use std::fs;
use std::time::Instant;

fn best_of<F: FnMut() -> usize>(mut f: F, reps: usize) -> (f64, usize) {
    let n = f(); // warm
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        let _ = f();
        best = best.min(t.elapsed().as_secs_f64());
    }
    (best, n)
}

fn read_ids(path: &str) -> Vec<u32> {
    let b = fs::read(path).expect("read ref ids");
    let n = u32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
    (0..n).map(|i| u32::from_le_bytes(b[4 + i * 4..8 + i * 4].try_into().unwrap())).collect()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let (path, model, ids_path) = (
        args.next().expect("usage: <corpus> <cl100k|o200k> <ref.ids>"),
        args.next().expect("model"),
        args.next().expect("ref.ids"),
    );
    let text = String::from_utf8_lossy(&fs::read(&path).expect("read corpus")).into_owned();
    let mb = text.len() as f64 / 1e6;
    let refids = read_ids(&ids_path);

    let (bpe, tik) = match model.as_str() {
        "o200k" => (bpe_openai::o200k_base(), tiktoken_rs::o200k_base().unwrap()),
        _ => (bpe_openai::cl100k_base(), tiktoken_rs::cl100k_base().unwrap()),
    };

    let bpe_ids: Vec<u32> = bpe.encode(&text);
    let tik_ids: Vec<u32> = tik.encode_ordinary(&text).iter().map(|&x| x as u32).collect();
    let bpe_ok = bpe_ids == refids;
    let tik_ok = tik_ids == refids;
    println!("  exact vs reference: bpe-openai {}  tiktoken-rs {}",
             if bpe_ok { "OK" } else { "*** MISMATCH ***" },
             if tik_ok { "OK" } else { "*** MISMATCH ***" });

    let (s_bpe, _) = best_of(|| bpe.encode(&text).len(), 5);
    let (s_tik, _) = best_of(|| tik.encode_ordinary(&text).len(), 5);
    println!("bpe-openai   {:8.2} MB/s", mb / s_bpe);
    println!("tiktoken-rs  {:8.2} MB/s", mb / s_tik);
    if !(bpe_ok && tik_ok) {
        std::process::exit(1);
    }
}
