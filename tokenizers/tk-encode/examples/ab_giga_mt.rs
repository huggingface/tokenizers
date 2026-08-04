//! Our side of the gigatoken multi-thread A/B: one 4 MB document per thread through the
//! parallel `encode`, matching gigatoken's `hf_mt` (`encode_docs_ragged` over the same
//! docs) in thread count, work per thread and MiB/s convention.
//!
//!   AB_TOKENIZER=<tokenizer.json> AB_CORPUS=<file> AB_THREADS=<n> AB_PASSES=<n>
//!
//! Prints the best pass, so the caches are warm -- the same face `hf_mt` reports.
use std::convert::TryFrom;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use tk_encode::Tokenizer;
use tk_encode::pipeline::PipelineTokenizer;

const DOC: usize = 4 * 1024 * 1024;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(default)
}

fn main() {
    let tok = PathBuf::from(std::env::var("AB_TOKENIZER").expect("AB_TOKENIZER"));
    let corpus = PathBuf::from(std::env::var("AB_CORPUS").expect("AB_CORPUS"));
    let threads = env_usize("AB_THREADS", 1);
    let passes = env_usize("AB_PASSES", 4);
    // Our pool reads this; set it before the first encode so the pool is built at this size.
    tk_encode::utils::parallelism::set_num_threads(threads);

    let legacy = Tokenizer::from_file(&tok).expect("load");
    let pipe = PipelineTokenizer::try_from(&legacy).expect("pipeline");

    let text = std::fs::read_to_string(&corpus).expect("corpus");
    let one = text.repeat(DOC.div_ceil(text.len()));
    let docs: Vec<&str> = (0..threads).map(|_| one.as_str()).collect();
    let total = one.len() * threads;

    let mut best = 0.0f64;
    let mut tokens = 0usize;
    for _ in 0..passes {
        let start = Instant::now();
        let out = pipe.encode(&docs[..]).wait_for_completion().expect("encode");
        let mbs = total as f64 / start.elapsed().as_secs_f64() / (1024.0 * 1024.0);
        tokens = out.iter().map(Vec::len).sum();
        black_box(out);
        best = best.max(mbs);
    }
    println!(
        "{:<10} {threads:>2}t  {best:>7.0} MB/s  {tokens} tokens",
        corpus.file_stem().unwrap().to_string_lossy()
    );
}
