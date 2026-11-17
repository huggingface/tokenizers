//! The pipeline encode path, over napi.
//!
//! This is the whole Node surface. The classes that used to live beside it -- `Tokenizer`,
//! `Model`, `Normalizer`, `PreTokenizer`, `Processor`, `Decoder`, the trainers and their async
//! tasks -- wrapped the pre-v1 engine, which no longer exists: `DecoderWrapper`,
//! `PostProcessorWrapper`, `BpeBuilder`, `TrainerWrapper`, `Trainable` and `NormalizedString`
//! were all removed with it, and this crate had stopped compiling. Rather than port wrappers for
//! an engine that is gone, the binding now exposes the path that replaced it.

use std::ptr;

use napi::JsValue;
use napi::bindgen_prelude::*;
use napi_derive::napi;
use tokenizers::pipeline::{self, Override};
use tokenizers::{
  PaddingDirection, PaddingParams, PaddingStrategy, TruncationDirection, TruncationParams,
  TruncationStrategy,
};

fn err<E: std::fmt::Display>(e: E) -> Error {
  Error::from_reason(format!("{e}"))
}

/// Refuses a typed array whose memory is a `SharedArrayBuffer`.
///
/// Another thread can write to a `SharedArrayBuffer` at any time, so Rust cannot hold a `&str` or
/// a `&mut [u32]` over it.
fn refuse_shared_buffer(typed_array: &Unknown, name: &str) -> Result<()> {
  let typed_array = typed_array.value();
  let mut buffer = ptr::null_mut();
  let mut is_array_buffer = false;
  // SAFETY: the env and the value come from an argument of the running call. napi skips an
  // out-pointer that is null, and the two others point to locals.
  unsafe {
    check_status!(
      sys::napi_get_typedarray_info(
        typed_array.env,
        typed_array.value,
        ptr::null_mut(),
        ptr::null_mut(),
        ptr::null_mut(),
        &mut buffer,
        ptr::null_mut(),
      ),
      "`{name}` is not a typed array"
    )?;
    // False for a `SharedArrayBuffer`, the only other buffer a typed array can sit on.
    check_status!(sys::napi_is_arraybuffer(
      typed_array.env,
      buffer,
      &mut is_array_buffer
    ))?;
  }
  if !is_array_buffer {
    return Err(err(format!(
      "`{name}` is a view of a SharedArrayBuffer, which another thread could write to during the call: pass a copy"
    )));
  }
  Ok(())
}

/// Per-call settings. A field left out keeps the tokenizer's own behaviour.
#[napi(object)]
#[derive(Default)]
pub struct EncodeOptions {
  /// Whether the post-processor adds its special tokens, such as `[CLS]` and `[SEP]`. `true`
  /// when left out.
  pub add_special_tokens: Option<bool>,
  /// Whether a special token written in the text goes through the model (`true`) or becomes its
  /// added-vocabulary id. `false` when left out.
  pub encode_special_tokens: Option<bool>,
  /// Padding for this call, replacing the tokenizer's configured padding. `false` disables
  /// padding. Left out, the configured padding applies.
  #[napi(ts_type = "false | PaddingOptions")]
  pub padding: Option<Either<bool, PaddingOptions>>,
  /// Truncation for this call, replacing the tokenizer's configured truncation. `false` disables
  /// truncation. Left out, the configured truncation applies.
  #[napi(ts_type = "false | TruncationOptions")]
  pub truncation: Option<Either<bool, TruncationOptions>>,
}

fn override_with<O, T>(
  option: Option<Either<bool, O>>,
  params: fn(O) -> Result<T>,
) -> Result<Override<T>> {
  Ok(match option {
    // `true` is outside the TS type; untyped callers passing it get the same as leaving it out.
    None | Some(Either::A(true)) => Override::InheritConfig,
    Some(Either::A(false)) => Override::Off,
    Some(Either::B(options)) => Override::With(params(options)?),
  })
}

fn padding_direction(name: &str) -> Result<PaddingDirection> {
  match name {
    "left" => Ok(PaddingDirection::Left),
    "right" => Ok(PaddingDirection::Right),
    other => Err(err(format!(
      "padding direction must be 'left' or 'right', not {other:?}"
    ))),
  }
}

/// Padding for one call. It replaces the tokenizer's configured padding as a whole, so a field
/// left out takes the value noted on the field, not the configured one.
#[napi(object)]
pub struct PaddingOptions {
  /// `right` when left out, or `left`: whether padding tokens are appended to the right or
  /// prepended to the left of encoded tokens.
  #[napi(ts_type = "'left' | 'right'")]
  pub direction: Option<String>,
  /// The id of the padding token. `0` when left out.
  pub pad_id: Option<u32>,
  /// The type id of the padding token. `0` when left out.
  pub pad_type_id: Option<u32>,
  /// The text of the padding token. `[PAD]` when left out.
  pub pad_token: Option<String>,
  /// Pads every encoding to exactly this many tokens. Left out, pads each batch to its longest
  /// item.
  pub length: Option<u32>,
  /// Rounds the padded length up to a multiple of this.
  pub pad_to_multiple_of: Option<u32>,
}

impl PaddingOptions {
  fn params(self) -> Result<PaddingParams> {
    let mut params = PaddingParams::default();
    params.strategy = self.length.map_or(PaddingStrategy::BatchLongest, |n| {
      PaddingStrategy::Fixed(n as usize)
    });
    params.direction = match self.direction {
      Some(name) => padding_direction(&name)?,
      None => PaddingDirection::Right,
    };
    params.pad_to_multiple_of = self.pad_to_multiple_of.map(|n| n as usize);
    params.pad_id = self.pad_id.unwrap_or(0);
    params.pad_type_id = self.pad_type_id.unwrap_or(0);
    params.pad_token = self.pad_token.unwrap_or_else(|| "[PAD]".to_owned());
    Ok(params)
  }
}

fn truncation_direction(name: &str) -> Result<TruncationDirection> {
  match name {
    "left" => Ok(TruncationDirection::Left),
    "right" => Ok(TruncationDirection::Right),
    other => Err(err(format!(
      "truncation direction must be 'left' or 'right', not {other:?}"
    ))),
  }
}

fn truncation_strategy(name: &str) -> Result<TruncationStrategy> {
  match name {
    "longest_first" => Ok(TruncationStrategy::LongestFirst),
    "only_first" => Ok(TruncationStrategy::OnlyFirst),
    "only_second" => Ok(TruncationStrategy::OnlySecond),
    other => Err(err(format!(
      "truncation strategy must be 'longest_first', 'only_first' or 'only_second', not {other:?}"
    ))),
  }
}

/// Truncation for one call. It replaces the tokenizer's configured truncation as a whole, so a
/// field left out takes the value noted on the field, not the configured one.
#[napi(object)]
pub struct TruncationOptions {
  /// The maximum number of tokens, including special tokens, to keep. Encodings with more tokens
  /// get truncated.
  pub max_length: u32,
  /// `longest_first` when left out, `only_first` or `only_second`: which sequence of a pair is
  /// truncated.
  #[napi(ts_type = "'longest_first' | 'only_first' | 'only_second'")]
  pub strategy: Option<String>,
  /// `right` when left out, or `left`: whether to truncate tokens at the end of the sequence or
  /// at its beginning.
  #[napi(ts_type = "'left' | 'right'")]
  pub direction: Option<String>,
}

impl TruncationOptions {
  fn params(self) -> Result<TruncationParams> {
    let mut params = TruncationParams::default();
    params.max_length = self.max_length as usize;
    params.strategy = match self.strategy {
      Some(name) => truncation_strategy(&name)?,
      None => TruncationStrategy::LongestFirst,
    };
    params.direction = match self.direction {
      Some(name) => truncation_direction(&name)?,
      None => TruncationDirection::Right,
    };
    // The pipeline keeps only the first window, so a stride changes nothing and is not exposed.
    params.stride = 0;
    Ok(params)
  }
}

impl PipelineTokenizer {
  fn encode_options(options: Option<EncodeOptions>) -> Result<pipeline::EncodeOptions> {
    let options = options.unwrap_or_default();
    let mut encode_options = pipeline::EncodeOptions::default();
    encode_options.add_special_tokens = options.add_special_tokens.unwrap_or(true);
    encode_options.encode_special_tokens = options.encode_special_tokens.unwrap_or(false);
    encode_options.padding = override_with(options.padding, PaddingOptions::params)?;
    encode_options.truncation = override_with(options.truncation, TruncationOptions::params)?;
    Ok(encode_options)
  }
}

#[napi]
pub struct PipelineTokenizer(pipeline::Tokenizer);

#[napi]
impl PipelineTokenizer {
  /// Read a `tokenizer.json`. The file is put through the legacy "1.0" -> canonical "2.0"
  /// upgrade first, so the tokenizers already on disk keep loading; `tk_serialize` itself only
  /// reads the canonical form.
  #[napi(factory)]
  pub fn from_file(path: String) -> Result<Self> {
    let canonical = tokenizers::canonicalize_file(&path).map_err(err)?;
    Ok(Self(tokenizers::from_json(&canonical).map_err(err)?))
  }

  /// `Uint32Array`, not `Vec<u32>`: a JS `Array` costs one napi value per token, which on
  /// token-dense input is 13x the encode itself (gpt2 chinese 31 vs 616 MB/s).
  #[napi]
  pub fn encode(&self, text: String, options: Option<EncodeOptions>) -> Result<Uint32Array> {
    let encodings = self
      .0
      .encode(text.as_str(), &Self::encode_options(options)?)
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
  ///
  /// Throws on a view of a `SharedArrayBuffer`, which another thread could write to during the
  /// call.
  #[napi(
    ts_args_type = "text: Uint8Array, out: Uint32Array, options?: EncodeOptions | undefined | null"
  )]
  pub fn encode_bytes_into(
    &self,
    text: Unknown,
    out: Unknown,
    options: Option<EncodeOptions>,
  ) -> Result<u32> {
    let options = Self::encode_options(options)?;
    // napi read `options` before this body, which ran its getters. A getter can detach or resize
    // a buffer, so the two arrays' pointers are read only now, and no JS runs from here on.
    refuse_shared_buffer(&text, "text")?;
    refuse_shared_buffer(&out, "out")?;
    let text = Uint8Array::from_unknown(text)?;
    let mut out = Uint32Array::from_unknown(out)?;
    // `text` is a handle, not a `&[u8]` argument, so its borrow ends with this block.
    let encodings = {
      let text = std::str::from_utf8(&text).map_err(err)?;
      self.0.encode(text, &options).wait().map_err(err)?
    };
    let ids = encodings.first().map(|e| e.ids()).unwrap_or(&[]);
    // SAFETY: `out` is not on a `SharedArrayBuffer` and no JS has run since it was read, so only
    // this call accesses its memory. The borrow of `text`, maybe a view of the same buffer, has ended.
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
