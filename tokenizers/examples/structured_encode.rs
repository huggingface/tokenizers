//! Encode a structured JSON input against a tokenizer config (legacy or canonical).
//! cargo run --example structured_encode -- tokenizer.json segments.json
use std::{env, fs};
use tokenizers::{canonicalize_file, from_json, segments_from_json};

fn main() -> tokenizers::Result<()> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 3 {
        return Err("usage: structured_encode TOKENIZER_JSON SEGMENTS_JSON".into());
    }
    let tokenizer = from_json(&canonicalize_file(&args[1])?)?;
    let segments = segments_from_json(&fs::read_to_string(&args[2])?)?;
    let encodings = tokenizer.encode_segments(&segments, true).wait()?;
    let ids: Vec<Vec<u32>> = encodings
        .iter()
        .map(|encoding| encoding.ids().iter().map(|token| token.id()).collect())
        .collect();
    println!("{}", serde_json::to_string(&ids)?);
    Ok(())
}
