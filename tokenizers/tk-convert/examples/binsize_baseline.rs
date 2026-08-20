//! Minimal encode program measured by CI for binary size: the latest released
//! `tokenizers` crate (the comparison baseline). Structurally identical to
//! `binsize_pipeline.rs`.

use tokenizers_release::Tokenizer;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: binsize_baseline <tokenizer.json> <text>");
    let text = args
        .next()
        .expect("usage: binsize_baseline <tokenizer.json> <text>");
    let tok = Tokenizer::from_file(path).unwrap();
    // `encode_fast` (offset-free) matches the pipeline's encode path — the path
    // the throughput benchmark times on both sides — so the sizes compare like for like.
    println!("{}", tok.encode_fast(text.as_str(), false).unwrap().len());
}
