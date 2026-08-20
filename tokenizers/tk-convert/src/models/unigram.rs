//! The config half of `Unigram`: reading and writing a bare `unigram.json`.
//!
//! Both are serde over the whole model, and the shape is the model's own (`tk-encode`'s
//! `models::unigram::serialization`). These two just point it at a file.

use std::fs::read_to_string;
use std::path::{Path, PathBuf};

use tk_encode::Result;
use tk_encode::models::unigram::Unigram;

/// Load a SentencePiece-style `unigram.json`, as `tokenizers` writes it.
///
/// ```no_run
/// let model = tk_convert::models::unigram::load("mymodel-unigram.json").unwrap();
/// ```
pub fn load<P: AsRef<Path>>(path: P) -> Result<Unigram> {
    let string = read_to_string(path)?;
    Ok(serde_json::from_str(&string)?)
}

/// Write the whole model as a pretty-printed `unigram.json`.
///
/// This is the body of what used to be `impl Model for Unigram`'s `save`, reached now through
/// [`ModelWrapper::save`](super::ModelWrapper) — see the note there for why a trait method's body
/// ended up on this side of the crate boundary.
pub fn save(model: &Unigram, folder: &Path, name: Option<&str>) -> Result<Vec<PathBuf>> {
    let name = match name {
        Some(name) => format!("{name}-unigram.json"),
        None => "unigram.json".to_string(),
    };
    let mut fullpath = PathBuf::new();
    fullpath.push(folder);
    fullpath.push(name);
    let string = serde_json::to_string_pretty(model)?;
    std::fs::write(&fullpath, string)?;
    Ok(vec![fullpath])
}
