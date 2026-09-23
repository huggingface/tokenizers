//! This crate allows to pass plain javascript objects as arguments to Tokenizer.encode
//! 
//! The wasm_bindgen macro generates code to store the struct in the WebAssembly memory (outside of the JS space).
//! 
//! For a simpler UX, this code declares plain TypeScript interfaces that can be passed directly to Tokenizer.encode()
//! instead of instantiating (and taking care of freeing) a wasm object.

use tk_encode::pipeline::{self, Override};
use tk_encode::{
    PaddingDirection, PaddingParams, PaddingStrategy, TruncationDirection, TruncationParams,
    TruncationStrategy,
};
use wasm_bindgen::prelude::*;

#[wasm_bindgen(typescript_custom_section)]
const OPTIONS: &str = r#"
/** Per-call settings. A field left out keeps the tokenizer's own behaviour. */
export interface EncodeOptions {
  /**
   * Whether the post-processor adds its special tokens, such as `[CLS]` and `[SEP]`. `true`
   * when left out.
   */
  addSpecialTokens?: boolean
  /**
   * Whether a special token written in the text goes through the model (`true`) or becomes its
   * added-vocabulary id. `false` when left out.
   */
  encodeSpecialTokens?: boolean
  /**
   * Padding for this call, replacing the tokenizer's configured padding. `false` disables
   * padding. Left out, the configured padding applies.
   */
  padding?: false | PaddingOptions
  /**
   * Truncation for this call, replacing the tokenizer's configured truncation. `false` disables
   * truncation. Left out, the configured truncation applies.
   */
  truncation?: false | TruncationOptions
}

/**
 * Padding for one call. It replaces the tokenizer's configured padding as a whole, so a field
 * left out takes the value noted on the field, not the configured one.
 */
export interface PaddingOptions {
  /**
   * `right` when left out, or `left`: whether padding tokens are appended to the right or
   * prepended to the left of encoded tokens.
   */
  direction?: 'left' | 'right'
  /** The id of the padding token. `0` when left out. */
  padId?: number
  /** The type id of the padding token. `0` when left out. */
  padTypeId?: number
  /** The text of the padding token. `[PAD]` when left out. */
  padToken?: string
  /**
   * Pads every encoding to exactly this many tokens. Left out, pads each batch to its longest
   * item.
   */
  length?: number
  /** Rounds the padded length up to a multiple of this. */
  padToMultipleOf?: number
}

/**
 * Truncation for one call. It replaces the tokenizer's configured truncation as a whole, so a
 * field left out takes the value noted on the field, not the configured one.
 */
export interface TruncationOptions {
  /**
   * The maximum number of tokens, including special tokens, to keep. Encodings with more tokens
   * get truncated.
   */
  maxLength: number
  /**
   * `longest_first` when left out, `only_first` or `only_second`: which sequence of a pair is
   * truncated.
   */
  strategy?: 'longest_first' | 'only_first' | 'only_second'
  /**
   * `right` when left out, or `left`: whether to truncate tokens at the end of the sequence or
   * at its beginning.
   */
  direction?: 'left' | 'right'
}
"#;


/// This block allows rust / wasm to read from plain javascript objects
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(typescript_type = "EncodeOptions")]
    pub type EncodeOptions;

    #[wasm_bindgen(method, getter, js_name = addSpecialTokens)]
    pub fn add_special_tokens(this: &EncodeOptions) -> Option<bool>;

    #[wasm_bindgen(method, getter, js_name = encodeSpecialTokens)]
    pub fn encode_special_tokens(this: &EncodeOptions) -> Option<bool>;

    #[wasm_bindgen(method, getter)]
    pub fn padding(this: &EncodeOptions) -> JsValue;

    #[wasm_bindgen(method, getter)]
    pub fn truncation(this: &EncodeOptions) -> JsValue;

    #[wasm_bindgen(typescript_type = "PaddingOptions")]
    pub type PaddingOptions;

    #[wasm_bindgen(method, getter)]
    pub fn direction(this: &PaddingOptions) -> Option<String>;

    #[wasm_bindgen(method, getter, js_name = padId)]
    pub fn pad_id(this: &PaddingOptions) -> Option<u32>;

    #[wasm_bindgen(method, getter, js_name = padTypeId)]
    pub fn pad_type_id(this: &PaddingOptions) -> Option<u32>;

    #[wasm_bindgen(method, getter, js_name = padToken)]
    pub fn pad_token(this: &PaddingOptions) -> Option<String>;

    #[wasm_bindgen(method, getter)]
    pub fn length(this: &PaddingOptions) -> Option<u32>;

    #[wasm_bindgen(method, getter, js_name = padToMultipleOf)]
    pub fn pad_to_multiple_of(this: &PaddingOptions) -> Option<u32>;

    #[wasm_bindgen(typescript_type = "TruncationOptions")]
    pub type TruncationOptions;

    #[wasm_bindgen(method, getter, js_name = maxLength)]
    pub fn max_length(this: &TruncationOptions) -> u32;

    #[wasm_bindgen(method, getter)]
    pub fn strategy(this: &TruncationOptions) -> Option<String>;

    #[wasm_bindgen(method, getter)]
    pub fn direction(this: &TruncationOptions) -> Option<String>;
}

/// Convert the local EncodeOptions wrapper to tk_encode::pipeline::EncodeOptions
pub(crate) fn convert_encode_options(
    options: Option<EncodeOptions>,
) -> Result<pipeline::EncodeOptions, JsError> {
    let options = options.as_ref();
    Ok(pipeline::EncodeOptions {
        add_special_tokens: options
            .and_then(EncodeOptions::add_special_tokens)
            .unwrap_or(true),
        encode_special_tokens: options
            .and_then(EncodeOptions::encode_special_tokens)
            .unwrap_or(false),
        padding: override_with(options.map(EncodeOptions::padding), PaddingOptions::params)?,
        truncation: override_with(
            options.map(EncodeOptions::truncation),
            TruncationOptions::params,
        )?,
    })
}

/// Parse a JS value into a Override
fn override_with<O: JsCast, T>(
    value: Option<JsValue>,
    params: fn(&O) -> Result<T, JsError>,
) -> Result<Override<T>, JsError> {
    let Some(value) = value else {
        return Ok(Override::InheritConfig);
    };
    Ok(match value.as_bool() {
        Some(false) => Override::Off,
        Some(true) => Override::InheritConfig,
        None if value.is_undefined() || value.is_null() => Override::InheritConfig,
        None => Override::With(params(value.unchecked_ref())?),
    })
}

impl PaddingOptions {
    pub(crate) fn params(&self) -> Result<PaddingParams, JsError> {
        Ok(PaddingParams {
            strategy: self
                .length()
                .map_or(PaddingStrategy::BatchLongest, |length| {
                    PaddingStrategy::Fixed(length as usize)
                }),
            direction: match self.direction().as_deref() {
                None | Some("right") => PaddingDirection::Right,
                Some("left") => PaddingDirection::Left,
                Some(other) => {
                    return Err(JsError::new(&format!(
                        "padding direction must be 'left' or 'right', not {other:?}"
                    )));
                }
            },
            pad_to_multiple_of: self.pad_to_multiple_of().map(|multiple| multiple as usize),
            pad_id: self.pad_id().unwrap_or(0),
            pad_type_id: self.pad_type_id().unwrap_or(0),
            pad_token: self.pad_token().unwrap_or_else(|| "[PAD]".to_owned()),
        })
    }
}

impl TruncationOptions {
    pub(crate) fn params(&self) -> Result<TruncationParams, JsError> {
        Ok(TruncationParams {
            max_length: self.max_length() as usize,
            strategy: match self.strategy().as_deref() {
                None | Some("longest_first") => TruncationStrategy::LongestFirst,
                Some("only_first") => TruncationStrategy::OnlyFirst,
                Some("only_second") => TruncationStrategy::OnlySecond,
                Some(other) => {
                    return Err(JsError::new(&format!(
                        "truncation strategy must be 'longest_first', 'only_first' or 'only_second', not {other:?}"
                    )));
                }
            },
            direction: match self.direction().as_deref() {
                None | Some("right") => TruncationDirection::Right,
                Some("left") => TruncationDirection::Left,
                Some(other) => {
                    return Err(JsError::new(&format!(
                        "truncation direction must be 'left' or 'right', not {other:?}"
                    )));
                }
            },
            stride: 0,
        })
    }
}

#[cfg(all(test, target_arch = "wasm32"))]
mod tests {
    use super::*;
    use pipeline::Override::{Off, With};
    use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};

    wasm_bindgen_test_configure!(run_in_browser);

    // `JSON.parse` builds the same plain object a caller would write, with the JavaScript names.
    fn parse_encode_options(json: &str) -> Result<pipeline::EncodeOptions, JsValue> {
        let options = js_sys::JSON::parse(json).unwrap().unchecked_into();
        convert_encode_options(Some(options)).map_err(JsValue::from)
    }

    #[wasm_bindgen_test]
    fn test_empty_is_default() {
        assert_eq!(parse_encode_options("{}").unwrap(), pipeline::EncodeOptions::default());
    }

    #[wasm_bindgen_test]
    fn test_camel_case() {
        let json = r#"{
            "addSpecialTokens": false,
            "encodeSpecialTokens": true,
            "padding": {"direction": "left", "length": 24, "padId": 1, "padTypeId": 2, "padToken": "<pad>", "padToMultipleOf": 8},
            "truncation": {"maxLength": 8, "strategy": "only_second", "direction": "left"}
        }"#;
        assert_eq!(
            parse_encode_options(json).unwrap(),
            pipeline::EncodeOptions {
                add_special_tokens: false,
                encode_special_tokens: true,
                padding: With(PaddingParams {
                    strategy: PaddingStrategy::Fixed(24),
                    direction: PaddingDirection::Left,
                    pad_to_multiple_of: Some(8),
                    pad_id: 1,
                    pad_type_id: 2,
                    pad_token: "<pad>".to_owned(),
                }),
                truncation: With(TruncationParams {
                    max_length: 8,
                    strategy: TruncationStrategy::OnlySecond,
                    direction: TruncationDirection::Left,
                    stride: 0,
                }),
            }
        );
    }

    #[wasm_bindgen_test]
    fn test_padding_truncation_false() {
        assert_eq!(
            parse_encode_options(r#"{"padding": false, "truncation": false}"#).unwrap(),
            pipeline::EncodeOptions {
                padding: Off,
                truncation: Off,
                ..Default::default()
            }
        );
    }

    #[wasm_bindgen_test]
    fn test_omitted_field_is_default() {
        assert_eq!(
            parse_encode_options(r#"{"padding": {}, "truncation": {"maxLength": 8}}"#).unwrap(),
            pipeline::EncodeOptions {
                padding: With(PaddingParams {
                    strategy: PaddingStrategy::BatchLongest,
                    direction: PaddingDirection::Right,
                    pad_to_multiple_of: None,
                    pad_id: 0,
                    pad_type_id: 0,
                    pad_token: "[PAD]".to_owned(),
                }),
                truncation: With(TruncationParams {
                    max_length: 8,
                    strategy: TruncationStrategy::LongestFirst,
                    direction: TruncationDirection::Right,
                    stride: 0,
                }),
                ..Default::default()
            }
        );
    }

    #[wasm_bindgen_test]
    fn test_reject_invalid_enum_value() {
        for json in [
            r#"{"padding": {"direction": "up"}}"#,
            r#"{"truncation": {"maxLength": 8, "direction": "up"}}"#,
            r#"{"truncation": {"maxLength": 8, "strategy": "shortest"}}"#,
        ] {
            assert!(parse_encode_options(json).is_err(), "{json}");
        }
    }
}
