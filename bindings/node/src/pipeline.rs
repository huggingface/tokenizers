//! The pipeline encode path, over napi.
//!
//! This is the whole Node surface. The classes that used to live beside it -- `Tokenizer`,
//! `Model`, `Normalizer`, `PreTokenizer`, `Processor`, `Decoder`, the trainers and their async
//! tasks -- wrapped the pre-v1 engine, which no longer exists: `DecoderWrapper`,
//! `PostProcessorWrapper`, `BpeBuilder`, `TrainerWrapper`, `Trainable` and `NormalizedString`
//! were all removed with it, and this crate had stopped compiling. Rather than port wrappers for
//! an engine that is gone, the binding now exposes the path that replaced it.

use napi::bindgen_prelude::*;
use napi_derive::napi;
use tk_encode::tokenizer::pipeline::PipelineTokenizer as Pipeline;

fn err<E: std::fmt::Display>(e: E) -> Error {
  Error::from_reason(format!("{e}"))
}

#[napi]
pub struct PipelineTokenizer(Pipeline);

#[napi]
impl PipelineTokenizer {
  /// Read a `tokenizer.json`. The file is put through the legacy "1.0" -> canonical "2.0"
  /// upgrade first, so the tokenizers already on disk keep loading; `tk_serialize` itself only
  /// reads the canonical form.
  #[napi(factory)]
  pub fn from_file(path: String) -> Result<Self> {
    let canonical = tk_convert::canonicalize_file(&path).map_err(err)?;
    Ok(Self(tk_serialize::from_json(&canonical).map_err(err)?))
  }

  /// `Uint32Array`, not `Vec<u32>`: a JS `Array` costs one napi value per token, which on
  /// token-dense input is 13x the encode itself (gpt2 chinese 31 vs 616 MB/s).
  #[napi]
  pub fn encode(&self, text: String, add_special_tokens: Option<bool>) -> Result<Uint32Array> {
    let encodings = self
      .0
      .encode(text.as_str(), add_special_tokens.unwrap_or(true))
      .wait()
      .map_err(err)?;
    let ids = encodings
      .first()
      .map(|e| e.ids().iter().map(|t| t.id()).collect())
      .unwrap_or_default();
    Ok(Uint32Array::new(ids))
  }

  /// Drops the two remaining per-call costs of [`Self::encode`]: the JS string -> UTF-8 copy
  /// (as fast as the tokenizer itself, so it halves throughput) and the fresh `ArrayBuffer`
  /// (388 ns of a 789 ns call). Returns how many ids were written.
  #[napi]
  pub fn encode_bytes_into(
    &self,
    text: &[u8],
    mut out: Uint32Array,
    add_special_tokens: Option<bool>,
  ) -> Result<u32> {
    let text = std::str::from_utf8(text).map_err(err)?;
    let encodings = self
      .0
      .encode(text, add_special_tokens.unwrap_or(true))
      .wait()
      .map_err(err)?;
    let ids = encodings.first().map(|e| e.ids()).unwrap_or(&[]);
    // SAFETY: JS is blocked for this synchronous call, so nothing else aliases `out`.
    let dst = unsafe { out.as_mut() };
    if ids.len() > dst.len() {
      return Err(err(format!(
        "need {} ids, buffer holds {}",
        ids.len(),
        dst.len()
      )));
    }
    for (d, t) in dst.iter_mut().zip(ids) {
      *d = t.id();
    }
    Ok(ids.len() as u32)
  }
}
