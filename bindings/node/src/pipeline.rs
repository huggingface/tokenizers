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
use tk_encode::pipeline::{EncodeOptions as PipelineEncodeOptions, Override};
use tk_encode::tokenizer::pipeline::PipelineTokenizer as Pipeline;
use tk_encode::{PaddingDirection, PaddingParams, PaddingStrategy};

fn err<E: std::fmt::Display>(e: E) -> Error {
  Error::from_reason(format!("{e}"))
}

/// Per-call settings. A field left out keeps the tokenizer's own behaviour.
#[napi(object)]
#[derive(Default)]
pub struct EncodeOptions {
  /// `true` unless set.
  pub add_special_tokens: Option<bool>,
  /// `false` turns the tokenizer's configured padding off, a `PaddingOptions` changes it for
  /// this call, `true` or left out keeps it.
  pub padding: Option<Either<bool, PaddingOptions>>,
}

fn direction(name: &str) -> Result<PaddingDirection> {
  match name {
    "left" => Ok(PaddingDirection::Left),
    "right" => Ok(PaddingDirection::Right),
    other => Err(err(format!(
      "padding direction is 'left' or 'right', not {other:?}"
    ))),
  }
}

/// A field left out keeps the tokenizer's configured padding, or the defaults when it configures
/// none: pad to the longest sequence in the batch, on the right, with id 0 and token `[PAD]`.
#[napi(object)]
pub struct PaddingOptions {
  /// Pad every sequence to this many tokens instead of to the longest one in the batch.
  pub length: Option<u32>,
  #[napi(ts_type = "'left' | 'right'")]
  pub direction: Option<String>,
  pub pad_to_multiple_of: Option<u32>,
  pub pad_id: Option<u32>,
  pub pad_type_id: Option<u32>,
  pub pad_token: Option<String>,
}

impl PaddingOptions {
  fn onto(self, configured: Option<&PaddingParams>) -> Result<PaddingParams> {
    let base = configured.cloned().unwrap_or_default();
    Ok(PaddingParams {
      strategy: self
        .length
        .map_or(base.strategy, |n| PaddingStrategy::Fixed(n as usize)),
      direction: match self.direction {
        Some(name) => direction(&name)?,
        None => base.direction,
      },
      pad_to_multiple_of: self
        .pad_to_multiple_of
        .map(|n| n as usize)
        .or(base.pad_to_multiple_of),
      pad_id: self.pad_id.unwrap_or(base.pad_id),
      pad_type_id: self.pad_type_id.unwrap_or(base.pad_type_id),
      pad_token: self.pad_token.unwrap_or(base.pad_token),
    })
  }
}

impl PipelineTokenizer {
  fn encode_options(&self, options: Option<EncodeOptions>) -> Result<PipelineEncodeOptions> {
    let options = options.unwrap_or_default();
    Ok(PipelineEncodeOptions {
      add_special_tokens: options.add_special_tokens.unwrap_or(true),
      padding: match options.padding {
        None | Some(Either::A(true)) => Override::InheritConfig,
        Some(Either::A(false)) => Override::Off,
        Some(Either::B(padding)) => Override::With(padding.onto(self.0.get_padding())?),
      },
    })
  }
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
  pub fn encode(&self, text: String, options: Option<EncodeOptions>) -> Result<Uint32Array> {
    let encodings = self
      .0
      .encode(text.as_str(), &self.encode_options(options)?)
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
    options: Option<EncodeOptions>,
  ) -> Result<u32> {
    let text = std::str::from_utf8(text).map_err(err)?;
    let encodings = self
      .0
      .encode(text, &self.encode_options(options)?)
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
