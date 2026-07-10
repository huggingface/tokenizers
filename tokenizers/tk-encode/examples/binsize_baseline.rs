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
    println!("{}", tok.encode(text.as_str(), false).unwrap().len());
}
