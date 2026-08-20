//! The config half of `WordLevel`: reading its `vocab.json`, and writing it back out.
//!
//! The model stays in `tk-encode` — it is a runtime model, taken straight through into the pipeline
//! — but both of these need serde and so cannot. `read_file` parses a `{token: id}` object with
//! `serde_json`, which is the single reason `serde_json` could not leave the runtime crate while it
//! was there; `save` writes the same object back through [`OrderedVocabIter`].

use ahash::AHashMap;
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};

use serde_json::Value;
use tk_encode::Result;
use tk_encode::models::wordlevel::{Error, WordLevel};

use tk_encode::models::OrderedVocabIter;

/// Read the given file to extract the vocab.
pub fn read_file(vocab_path: &str) -> Result<AHashMap<String, u32>> {
    let vocab_file = File::open(vocab_path)?;
    let mut vocab_file = BufReader::new(vocab_file);
    let mut buffer = String::new();
    let mut vocab = AHashMap::new();

    vocab_file.read_to_string(&mut buffer)?;
    let json: Value = serde_json::from_str(&buffer)?;

    match json {
        Value::Object(m) => {
            for (token, id) in m {
                if let Value::Number(id) = id {
                    let id = id.as_u64().ok_or(Error::BadVocabulary)? as u32;
                    vocab.insert(token, id);
                }
            }
        }
        _ => return Err(Box::new(Error::BadVocabulary)),
    };
    Ok(vocab)
}

/// Initialize a `WordLevel` model from a vocab file.
pub fn from_file(vocab_path: &str, unk_token: String) -> Result<WordLevel> {
    let vocab = read_file(vocab_path)?;
    WordLevel::builder()
        .vocab(vocab)
        .unk_token(unk_token)
        .build()
}

/// Write the model's vocabulary as a `vocab.json`.
///
/// This is the body of what used to be `impl Model for WordLevel`'s `save`. It is reached through
/// [`ModelWrapper::save`](super::ModelWrapper), which every real caller goes through; the impl left
/// behind in `tk-encode` says so, because a trait method cannot move across a crate boundary and
/// this one needs serde.
pub fn save(model: &WordLevel, folder: &Path, name: Option<&str>) -> Result<Vec<PathBuf>> {
    let vocab_file_name = match name {
        Some(name) => format!("{name}-vocab.json"),
        None => "vocab.json".to_string(),
    };

    // Write vocab.json
    let vocab_path: PathBuf = [folder, Path::new(vocab_file_name.as_str())]
        .iter()
        .collect();
    let mut vocab_file = File::create(&vocab_path)?;
    let order_vocab_iter = OrderedVocabIter::new(&model.vocab_r);
    let serialized = serde_json::to_string(&order_vocab_iter)?;
    vocab_file.write_all(serialized.as_bytes())?;

    Ok(vec![vocab_path])
}
