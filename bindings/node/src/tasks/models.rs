extern crate tokenizers as tk;

use crate::models::Model;
use napi::bindgen_prelude::*;
use std::sync::{Arc, RwLock};
use tokenizers::models::bpe::{BPE, BpeBuilder};
use tokenizers::models::wordlevel::{WordLevel, WordLevelBuilder};
use tokenizers::models::wordpiece::{WordPiece, WordPieceBuilder};

pub struct BPEFromFilesTask {
  pub(crate) builder: Option<BpeBuilder>,
}

impl Task for BPEFromFilesTask {
  type Output = BPE;
  type JsValue = Model;

  fn compute(&mut self) -> Result<Self::Output> {
    self
      .builder
      .take()
      .ok_or(Error::from_reason("Empty builder".to_string()))?
      .build()
      .map_err(|e| Error::from_reason(format!("{e}")))
  }

  fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
    Ok(Model {
      model: Some(Arc::new(RwLock::new(output.into()))),
    })
  }
}

pub struct WordPieceFromFilesTask {
  pub(crate) builder: Option<WordPieceBuilder>,
  /// The `vocab.txt` to load, read here in `compute` rather than eagerly by the caller: it used to
  /// be `WordPieceBuilder::files`, whose read happened inside `build()`, and that kept the file I/O
  /// off the JS thread. `files` moved to `tk-convert` with every other way of loading a vocabulary.
  pub(crate) vocab: String,
}

impl Task for WordPieceFromFilesTask {
  type Output = WordPiece;
  type JsValue = Model;

  fn compute(&mut self) -> Result<Self::Output> {
    let vocab = tk::models::wordpiece::read_file(&self.vocab)
      .map_err(|e| Error::from_reason(format!("{e}")))?;
    self
      .builder
      .take()
      .ok_or(Error::from_reason("Empty builder".to_string()))?
      .vocab(vocab)
      .build()
      .map_err(|e| Error::from_reason(format!("{e}")))
  }

  fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
    Ok(Model {
      model: Some(Arc::new(RwLock::new(output.into()))),
    })
  }
}
pub struct WordLevelFromFilesTask {
  pub(crate) builder: Option<WordLevelBuilder>,
  /// The `vocab.json` to load. Same reason as `WordPieceFromFilesTask::vocab` above.
  pub(crate) vocab: String,
}

impl Task for WordLevelFromFilesTask {
  type Output = WordLevel;
  type JsValue = Model;

  fn compute(&mut self) -> Result<Self::Output> {
    let vocab = tk::models::wordlevel::read_file(&self.vocab)
      .map_err(|e| Error::from_reason(format!("{e}")))?;
    self
      .builder
      .take()
      .ok_or(Error::from_reason("Empty builder".to_string()))?
      .vocab(vocab)
      .build()
      .map_err(|e| Error::from_reason(format!("{e}")))
  }

  fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
    Ok(Model {
      model: Some(Arc::new(RwLock::new(output.into()))),
    })
  }
}
