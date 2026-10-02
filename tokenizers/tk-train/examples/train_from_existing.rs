//! Trains a new tokenizer with every setting of an existing one, on new text, and saves it as a
//! `tokenizer.json`.
//!
//!   cargo run --release -p tk-train --example train_from_existing -- data/gpt2.json new.json 32000 corpus.txt [more.txt ...]
//!
//! The vocabulary is learned from scratch on the new text; nothing of the existing one is kept. The
//! normalizers, pre-tokenizer, post-processor, decoder, special and added tokens and model options
//! (unknown token, byte fallback, subword affixes) are copied over.

use tk_train::TokenizerTrainerBuilder;

fn main() -> tk_encode::Result<()> {
    let mut args = std::env::args().skip(1);
    let usage =
        "usage: train_from_existing <existing.json> <output.json> <vocab_size> <corpus.txt>...";
    let existing = args.next().expect(usage);
    let output = args.next().expect(usage);
    let vocab_size: usize = args.next().expect(usage).parse()?;
    let files: Vec<String> = args.collect();

    // `canonicalize_str` upgrades a `tokenizer.json` written by an older release.
    let existing = tk_serialize::from_json(&tk_convert::canonicalize_str(
        &std::fs::read_to_string(existing)?,
    )?)?;

    let tokenizer = TokenizerTrainerBuilder::from_tokenizer(&existing)?
        .vocab_size(vocab_size)
        .build()?
        .train_files(&files)?;

    std::fs::write(&output, tk_serialize::to_json(&tokenizer)?)?;
    Ok(())
}
