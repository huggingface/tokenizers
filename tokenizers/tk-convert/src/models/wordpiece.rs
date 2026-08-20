//! The config half of `WordPiece`: loading a vocabulary from a file, and building one from a `BPE`.
//!
//! The model itself stays in `tk-encode` — unlike `BPE` it *is* the runtime model, taken straight
//! through into the pipeline. What moved here is everything about getting one out of a file: a
//! `vocab.txt` is a config artifact, and `from_bytes` needs serde, which the runtime crate does not
//! link. They are free functions rather than associated ones because an inherent `impl` has to live
//! with the type.

use ahash::AHashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

use tk_encode::Result;
use tk_encode::models::wordpiece::{WordPiece, WordPieceBuilder};

use super::BPE;
use super::mirror;

/// Read the given file to extract the vocab: one token per line, id = line number.
pub fn read_file(vocab: &str) -> Result<AHashMap<String, u32>> {
    let file = File::open(vocab)?;
    read_bytes_inner(BufReader::new(file))
}

/// The same format, already in memory.
pub fn read_bytes(vocab: &[u8]) -> Result<AHashMap<String, u32>> {
    read_bytes_inner(BufReader::new(vocab))
}

fn read_bytes_inner<R: BufRead>(reader: R) -> Result<AHashMap<String, u32>> {
    let mut vocab = AHashMap::new();
    for (index, line) in reader.lines().enumerate() {
        let line = line?;
        vocab.insert(line.trim_end().to_owned(), index as u32);
    }

    Ok(vocab)
}

/// Initialize a `WordPiece` builder from a vocab mapping file.
pub fn from_file(vocab: &str) -> Result<WordPieceBuilder> {
    Ok(WordPiece::builder().vocab(read_file(vocab)?))
}

/// Read a `WordPiece` from its serialized form.
pub fn from_bytes<P: AsRef<[u8]>>(bytes: P) -> Result<WordPiece> {
    let mut de = serde_json::Deserializer::from_slice(bytes.as_ref());
    Ok(mirror::wordpiece::deserialize(&mut de)?)
}

/// Create a `WordPiece` model from a `BPE` model.
///
/// The one caller is the WordPiece trainer, which trains a `BPE` and then reinterprets its
/// vocabulary as WordPiece pieces. It names `BPE`, so it had to come along with it.
pub fn from_bpe(bpe: &BPE) -> WordPiece {
    let mut wp = WordPiece::builder()
        .vocab(bpe.get_vocab().into_iter().collect::<AHashMap<_, _>>())
        .build()
        .unwrap();
    if let Some(unk) = bpe.get_unk_token() {
        unk.clone_into(&mut wp.unk_token);
    }
    if let Some(prefix) = bpe.get_continuing_subword_prefix() {
        prefix.clone_into(&mut wp.continuing_subword_prefix);
    }
    wp
}
