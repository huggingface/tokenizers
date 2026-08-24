//! One tokenizer.json per model, fetched from the Hub and cached by `hf-hub`.
//!
//! This replaces the `make models` / `make fixtures` prerequisite: the oracles are self-contained,
//! and only the first run is online.

/// Repos covering the shapes the conversion has to handle: byte-level BPE (`ByteLevel`
/// pre-tokenizer), WordPiece, and SentencePiece Unigram (`Metaspace`).
pub const MODELS: &[&str] = &[
    "gpt2",
    "bert-base-uncased",
    "t5-base",
    "albert-base-v1",
    "meta-llama/Llama-3.2-1B",
];

/// Short inputs that still reach the paths the byte and delimiter handling live on: ASCII, accents,
/// CJK, and leading whitespace.
pub const TEXTS: &[&str] = &[
    "The quick brown fox jumps 123.",
    " héllo wörld ",
    "你好世界",
];

/// `None` when the file cannot be fetched, which is the offline case.
pub fn tokenizer_json(repo: &str) -> Option<std::path::PathBuf> {
    match hf_hub::api::sync::Api::new()
        .and_then(|api| api.model(repo.to_string()).get("tokenizer.json"))
    {
        Ok(path) => Some(path),
        Err(e) => {
            eprintln!("skip {repo}: cannot fetch tokenizer.json ({e})");
            None
        }
    }
}
