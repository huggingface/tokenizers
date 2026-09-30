//! The options of `Tokenizer.encode`

use serde::Deserialize;
use tk_encode::pipeline::{self, Override};
use tk_encode::{
    PaddingDirection, PaddingParams, PaddingStrategy, TruncationDirection, TruncationParams,
    TruncationStrategy,
};
use tsify::{Ts, Tsify};
use wasm_bindgen::prelude::*;

/// A convenience type to match typescript's `T | false` type
enum Setting<T> {
    Toggle(bool),
    With(T),
}

// We need to implement Deserialize ourselves to map typescript's `bool | T` type to rust [`Settings`]
impl<'de, T: Deserialize<'de>> Deserialize<'de> for Setting<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Visitor<T>(std::marker::PhantomData<T>);
        impl<'de, T: Deserialize<'de>> serde::de::Visitor<'de> for Visitor<T> {
            type Value = Setting<T>;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("boolean or an object")
            }
            fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
                Ok(Setting::Toggle(value))
            }
            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                map: A,
            ) -> Result<Self::Value, A::Error> {
                T::deserialize(serde::de::value::MapAccessDeserializer::new(map)).map(Setting::With)
            }
        }
        deserializer.deserialize_any(Visitor(std::marker::PhantomData))
    }
}

/// Override [`Tokenizer.encode`] settings. Omitted values default to the tokenizer's defaults / config
#[derive(Tsify, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct EncodeOptions {
    /// Whether the post-processor adds its special tokens, such as `[CLS]` and `[SEP]`.
    /// Defaults to `true` when omitted.
    #[tsify(optional)]
    add_special_tokens: Option<bool>,
    /// Whether a special token written in the text goes through the model (`true`) or becomes its
    /// added-vocabulary id.
    /// Defaults to `false` when omitted.
    #[tsify(optional)]
    encode_special_tokens: Option<bool>,
    /// Padding for this call, replacing the tokenizer's configured padding.
    /// `false` disables padding.
    /// Defaults to the tokenizer's config when omitted.
    #[tsify(optional, type = "false | PaddingOptions")]
    padding: Option<Setting<PaddingOptions>>,
    /// Truncation for this call, replacing the tokenizer's configured truncation.
    /// `false` disables truncation.
    /// Defaults to the tokenizer's config when omitted.
    #[tsify(optional, type = "false | TruncationOptions")]
    truncation: Option<Setting<TruncationOptions>>,
}

/// Padding for one call. It replaces the tokenizer's configured padding as a whole, so a field
/// left out takes the value noted on the field, not the configured one.
#[derive(Tsify, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PaddingOptions {
    /// `right` when left out, or `left`: whether padding tokens are appended to the right or
    /// prepended to the left of encoded tokens.
    #[tsify(optional)]
    direction: Option<Direction>,
    /// The id of the padding token. `0` when left out.
    #[tsify(optional)]
    pad_id: Option<u32>,
    /// The type id of the padding token. `0` when left out.
    #[tsify(optional)]
    pad_type_id: Option<u32>,
    /// The text of the padding token. `[PAD]` when left out.
    #[tsify(optional)]
    pad_token: Option<String>,
    /// Pads every encoding to exactly this many tokens. Left out, pads each batch to its longest
    /// item.
    #[tsify(optional)]
    length: Option<u32>,
    /// Rounds the padded length up to a multiple of this.
    #[tsify(optional)]
    pad_to_multiple_of: Option<u32>,
}

/// Truncation for one call. It replaces the tokenizer's configured truncation as a whole, so a
/// field left out takes the value noted on the field, not the configured one.
#[derive(Tsify, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TruncationOptions {
    /// The maximum number of tokens, including special tokens, to keep. Encodings with more tokens
    /// get truncated.
    max_length: u32,
    /// `longest_first` when left out, `only_first` or `only_second`: which sequence of a pair is
    /// truncated.
    #[tsify(optional)]
    strategy: Option<Strategy>,
    /// `right` when left out, or `left`: whether to truncate tokens at the end of the sequence or
    /// at its beginning.
    #[tsify(optional)]
    direction: Option<Direction>,
}

#[derive(Tsify, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Direction {
    Left,
    Right,
}

#[derive(Tsify, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Strategy {
    LongestFirst,
    OnlyFirst,
    OnlySecond,
}

pub(crate) fn convert_encode_options(
    options: Option<Ts<EncodeOptions>>,
) -> Result<pipeline::EncodeOptions, JsError> {
    let options = options
        .map(|options| options.to_rust())
        .transpose()?
        .unwrap_or_default();
    Ok(pipeline::EncodeOptions {
        add_special_tokens: options.add_special_tokens.unwrap_or(true),
        encode_special_tokens: options.encode_special_tokens.unwrap_or(false),
        padding: into_override(options.padding, PaddingOptions::params),
        truncation: into_override(options.truncation, TruncationOptions::params),
    })
}

fn into_override<O, T>(setting: Option<Setting<O>>, params: fn(O) -> T) -> Override<T> {
    match setting {
        None | Some(Setting::Toggle(true)) => Override::InheritConfig,
        Some(Setting::Toggle(false)) => Override::Off,
        Some(Setting::With(options)) => Override::With(params(options)),
    }
}

impl PaddingOptions {
    fn params(self) -> PaddingParams {
        PaddingParams {
            strategy: self.length.map_or(PaddingStrategy::BatchLongest, |length| {
                PaddingStrategy::Fixed(length as usize)
            }),
            direction: match self.direction {
                None | Some(Direction::Right) => PaddingDirection::Right,
                Some(Direction::Left) => PaddingDirection::Left,
            },
            pad_to_multiple_of: self.pad_to_multiple_of.map(|multiple| multiple as usize),
            pad_id: self.pad_id.unwrap_or(0),
            pad_type_id: self.pad_type_id.unwrap_or(0),
            pad_token: self.pad_token.unwrap_or_else(|| "[PAD]".to_owned()),
        }
    }
}

impl TruncationOptions {
    fn params(self) -> TruncationParams {
        TruncationParams {
            max_length: self.max_length as usize,
            strategy: match self.strategy {
                None | Some(Strategy::LongestFirst) => TruncationStrategy::LongestFirst,
                Some(Strategy::OnlyFirst) => TruncationStrategy::OnlyFirst,
                Some(Strategy::OnlySecond) => TruncationStrategy::OnlySecond,
            },
            direction: match self.direction {
                None | Some(Direction::Right) => TruncationDirection::Right,
                Some(Direction::Left) => TruncationDirection::Left,
            },
            stride: 0,
        }
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
        let options = Ts::new_unchecked(js_sys::JSON::parse(json).unwrap());
        convert_encode_options(Some(options)).map_err(JsValue::from)
    }

    #[wasm_bindgen_test]
    fn test_empty_is_default() {
        assert_eq!(
            parse_encode_options("{}").unwrap(),
            pipeline::EncodeOptions::default()
        );
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
    fn test_reject_invalid_type() {
        for json in [
            r#"{"addSpecialTokens": "no"}"#,
            r#"{"padding": 3}"#,
            r#"{"padding": {"padId": -1}}"#,
            r#"{"truncation": {"maxLength": "8"}}"#,
        ] {
            assert!(parse_encode_options(json).is_err(), "{json}");
        }
    }

    #[wasm_bindgen_test]
    fn test_reject_missing_max_length() {
        assert!(parse_encode_options(r#"{"truncation": {}}"#).is_err());
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
