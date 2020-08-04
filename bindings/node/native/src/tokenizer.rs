extern crate tokenizers as tk;

use crate::decoders::{Decoder, JsDecoder};
use crate::encoding::JsEncoding;
use crate::extraction::*;
use crate::models::{JsModel, Model};
use crate::normalizers::{JsNormalizer, Normalizer};
use crate::pre_tokenizers::{JsPreTokenizer, PreTokenizer};
use crate::processors::{JsPostProcessor, Processor};
use crate::tasks::tokenizer::{DecodeTask, EncodeTask};
use crate::trainers::JsTrainer;
use neon::prelude::*;
use std::sync::{Arc, RwLock};

use tk::TokenizerImpl;

// AddedToken

#[derive(Clone)]
pub struct AddedToken {
    pub token: tk::AddedToken,
}
impl From<AddedToken> for tk::AddedToken {
    fn from(v: AddedToken) -> Self {
        v.token
    }
}

#[allow(non_snake_case)]
#[derive(Debug, Default, Serialize, Deserialize)]
struct AddedTokenOptions {
    singleWord: Option<bool>,
    leftStrip: Option<bool>,
    rightStrip: Option<bool>,
    normalized: Option<bool>,
}
impl AddedTokenOptions {
    fn into_added_token(self, content: String, special: bool) -> tk::AddedToken {
        let mut token = tk::AddedToken::from(content, special);
        if let Some(sw) = self.singleWord {
            token = token.single_word(sw);
        }
        if let Some(ls) = self.leftStrip {
            token = token.lstrip(ls);
        }
        if let Some(rs) = self.rightStrip {
            token = token.rstrip(rs);
        }
        if let Some(n) = self.normalized {
            token = token.normalized(n);
        }
        token
    }
}

declare_types! {
    pub class JsAddedToken for AddedToken {
        init(mut cx) {
            // init(
            //  content: string,
            //  special: boolean,
            //  options?: {
            //    singleWord?: boolean = false,
            //    leftStrip?: boolean = false,
            //    rightStrip?: boolean = false
            //    normalized?: boolean = true,
            //  }
            // )

            let content = cx.extract::<String>(0)?;
            let special = cx.extract::<bool>(1)?;
            let token = cx.extract_opt::<AddedTokenOptions>(2)?
                .unwrap_or_else(AddedTokenOptions::default)
                .into_added_token(content, special);

            Ok(AddedToken { token })
        }

        method getContent(mut cx) {
            // getContent()

            let this = cx.this();
            let content = {
                let guard = cx.lock();
                let token = this.borrow(&guard);
                token.token.content.clone()
            };

            Ok(cx.string(content).upcast())
        }
    }
}

impl FromJsValue for AddedToken {
    fn from_value<'c, C: Context<'c>>(from: Handle<'c, JsValue>, cx: &mut C) -> LibResult<Self> {
        if let Ok(token) = from.downcast::<JsString>() {
            Ok(AddedToken {
                token: tk::AddedToken::from(token.value(), false),
            })
        } else if let Ok(token) = from.downcast::<JsAddedToken>() {
            let guard = cx.lock();
            let token = token.borrow(&guard);
            Ok(token.clone())
        } else {
            Err(Error("Expected `string | AddedToken`".into()))
        }
    }
}

struct SpecialToken(tk::AddedToken);
impl FromJsValue for SpecialToken {
    fn from_value<'c, C: Context<'c>>(from: Handle<'c, JsValue>, cx: &mut C) -> LibResult<Self> {
        if let Ok(token) = from.downcast::<JsString>() {
            Ok(SpecialToken(tk::AddedToken::from(token.value(), true)))
        } else if let Ok(token) = from.downcast::<JsAddedToken>() {
            let guard = cx.lock();
            let token = token.borrow(&guard);
            Ok(SpecialToken(token.token.clone()))
        } else {
            Err(Error("Expected `string | AddedToken`".into()))
        }
    }
}

// encode & encodeBatch types

struct TextInputSequence(tk::InputSequence);
struct PreTokenizedInputSequence(tk::InputSequence);
impl FromJsValue for PreTokenizedInputSequence {
    fn from_value<'c, C: Context<'c>>(from: Handle<'c, JsValue>, cx: &mut C) -> LibResult<Self> {
        let sequence = from
            .downcast::<JsArray>()?
            .to_vec(cx)?
            .into_iter()
            .map(|v| Ok(v.downcast::<JsString>()?.value()))
            .collect::<LibResult<Vec<_>>>()?;
        Ok(Self(sequence.into()))
    }
}
impl From<PreTokenizedInputSequence> for tk::InputSequence {
    fn from(v: PreTokenizedInputSequence) -> Self {
        v.0
    }
}
impl FromJsValue for TextInputSequence {
    fn from_value<'c, C: Context<'c>>(from: Handle<'c, JsValue>, _cx: &mut C) -> LibResult<Self> {
        Ok(Self(from.downcast::<JsString>()?.value().into()))
    }
}
impl From<TextInputSequence> for tk::InputSequence {
    fn from(v: TextInputSequence) -> Self {
        v.0
    }
}

struct TextEncodeInput(tk::EncodeInput);
struct PreTokenizedEncodeInput(tk::EncodeInput);
impl FromJsValue for PreTokenizedEncodeInput {
    fn from_value<'c, C: Context<'c>>(from: Handle<'c, JsValue>, cx: &mut C) -> LibResult<Self> {
        // If array is of size 2, and the first element is also an array, we'll parse a pair
        let array = from.downcast::<JsArray>()?;
        let is_pair = array.len() == 2
            && array
                .get(cx, 0)
                .map_or(false, |a| a.downcast::<JsArray>().is_ok());

        if is_pair {
            let first_seq: tk::InputSequence =
                PreTokenizedInputSequence::from_value(array.get(cx, 0)?, cx)?.into();
            let pair_seq: tk::InputSequence =
                PreTokenizedInputSequence::from_value(array.get(cx, 1)?, cx)?.into();
            Ok(Self((first_seq, pair_seq).into()))
        } else {
            Ok(Self(
                PreTokenizedInputSequence::from_value(from, cx)?.into(),
            ))
        }
    }
}
impl From<PreTokenizedEncodeInput> for tk::EncodeInput {
    fn from(v: PreTokenizedEncodeInput) -> Self {
        v.0
    }
}
impl FromJsValue for TextEncodeInput {
    fn from_value<'c, C: Context<'c>>(from: Handle<'c, JsValue>, cx: &mut C) -> LibResult<Self> {
        // If we get an array, it's a pair of sequences
        if let Ok(array) = from.downcast::<JsArray>() {
            if array.len() != 2 {
                return Err(Error(
                    "TextEncodeInput should be \
                    `TextInputSequence | [TextInputSequence, TextInputSequence]`"
                        .into(),
                ));
            }

            let first_seq: tk::InputSequence =
                TextInputSequence::from_value(array.get(cx, 0)?, cx)?.into();
            let pair_seq: tk::InputSequence =
                TextInputSequence::from_value(array.get(cx, 1)?, cx)?.into();
            Ok(Self((first_seq, pair_seq).into()))
        } else {
            Ok(Self(TextInputSequence::from_value(from, cx)?.into()))
        }
    }
}
impl From<TextEncodeInput> for tk::EncodeInput {
    fn from(v: TextEncodeInput) -> Self {
        v.0
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct EncodeOptions {
    #[serde(default)]
    is_pretokenized: bool,
    #[serde(default)]
    add_special_tokens: bool,
}
impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            is_pretokenized: false,
            add_special_tokens: true,
        }
    }
}

// Encoding

#[repr(transparent)]
pub struct Encoding(tk::Encoding);
impl FromJsValue for Encoding {
    fn from_value<'c, C: Context<'c>>(from: Handle<'c, JsValue>, cx: &mut C) -> LibResult<Self> {
        from.downcast::<JsEncoding>()
            .map(|e| {
                let guard = cx.lock();
                let enc = e.borrow(&guard);
                Self(enc.encoding.execute(|e| *e.unwrap().clone()))
            })
            .map_err(|_| Error("Expected Encoding".into()))
    }
}
impl From<Encoding> for tk::Encoding {
    fn from(v: Encoding) -> Self {
        v.0
    }
}

// Truncation

#[derive(Serialize, Deserialize)]
#[serde(remote = "tk::TruncationStrategy", rename_all = "snake_case")]
pub enum TruncationStrategyDef {
    LongestFirst,
    OnlyFirst,
    OnlySecond,
}

#[derive(Serialize, Deserialize)]
#[serde(
    remote = "tk::TruncationParams",
    rename_all = "camelCase",
    default = "tk::TruncationParams::default"
)]
pub struct TruncationParamsDef {
    max_length: usize,
    #[serde(with = "TruncationStrategyDef")]
    strategy: tk::TruncationStrategy,
    stride: usize,
}

#[derive(Serialize, Deserialize)]
#[serde(transparent)]
pub struct TruncationParams(#[serde(with = "TruncationParamsDef")] pub tk::TruncationParams);

// Padding

#[derive(Serialize, Deserialize)]
#[serde(remote = "tk::PaddingDirection", rename_all = "camelCase")]
pub enum PaddingDirectionDef {
    Left,
    Right,
}

// Here we define a custom method of serializing and deserializing a PaddingStrategy because
// we want it to actually be very different from the classic representation.
// In Rust, we use an enum to define the strategy, but in JS, we just want to have a optional
// length number => If defined we use the Fixed(n) strategy and otherwise the BatchLongest.
pub mod padding_strategy_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    #[derive(Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct Strategy {
        #[serde(skip_serializing_if = "Option::is_none")]
        max_length: Option<usize>,
    }

    pub fn serialize<S>(value: &tk::PaddingStrategy, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let s = Strategy {
            max_length: match value {
                tk::PaddingStrategy::BatchLongest => None,
                tk::PaddingStrategy::Fixed(s) => Some(*s),
            },
        };
        s.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<tk::PaddingStrategy, D::Error>
    where
        D: Deserializer<'de>,
    {
        let v = Strategy::deserialize(deserializer)?;
        if let Some(length) = v.max_length {
            Ok(tk::PaddingStrategy::Fixed(length))
        } else {
            Ok(tk::PaddingStrategy::BatchLongest)
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(
    remote = "tk::PaddingParams",
    rename_all = "camelCase",
    default = "tk::PaddingParams::default"
)]
pub struct PaddingParamsDef {
    #[serde(flatten, with = "padding_strategy_serde")]
    strategy: tk::PaddingStrategy,
    #[serde(with = "PaddingDirectionDef")]
    direction: tk::PaddingDirection,
    #[serde(skip_serializing_if = "Option::is_none")]
    pad_to_multiple_of: Option<usize>,
    pad_id: u32,
    pad_type_id: u32,
    pad_token: String,
}
#[derive(Serialize, Deserialize)]
#[serde(transparent)]
pub struct PaddingParams(#[serde(with = "PaddingParamsDef")] pub tk::PaddingParams);

type Tokenizer = TokenizerImpl<Model, Normalizer, PreTokenizer, Processor, Decoder>;

/// Tokenizer
#[derive(Clone)]
pub struct RsTokenizer {
    pub(crate) tokenizer: Arc<RwLock<Tokenizer>>,
}

declare_types! {
    pub class JsTokenizer for RsTokenizer {
        init(mut cx) {
            // init(model: JsModel)
            let model = cx.argument::<JsModel>(0)?;
            let guard = cx.lock();
            let model = model.borrow(&guard).clone();

            Ok(RsTokenizer {
                tokenizer: Arc::new(RwLock::new(TokenizerImpl::new(model)))
            })
        }

        method toString(mut cx) {
            // toString(pretty?: bool): string
            let pretty = cx.extract_opt::<bool>(0)?.unwrap_or(false);

            let this = cx.this();
            let guard = cx.lock();
            let s = this.borrow(&guard)
                .tokenizer.read().unwrap()
                .to_string(pretty)
                .map_err(|e| Error(format!("{}", e)))?;

            Ok(cx.string(s).upcast())
        }

        method save(mut cx) {
            // save(path: striing, pretty?: bool): undefined
            let path = cx.extract::<String>(0)?;
            let pretty = cx.extract_opt::<bool>(1)?.unwrap_or(false);

            let this = cx.this();
            let guard = cx.lock();
            this.borrow(&guard)
                .tokenizer.read().unwrap()
                .save(&path, pretty)
                .map_err(|e| Error(format!("{}", e)))?;

            Ok(cx.undefined().upcast())
        }

        method runningTasks(mut cx) {
            // runningTasks(): number
            let this = cx.this();
            let guard = cx.lock();
            let count = std::sync::Arc::strong_count(&this.borrow(&guard).tokenizer);
            let running = if count > 0 { count - 1 } else { 0 };
            Ok(cx.number(running as f64).upcast())
        }

        method getVocab(mut cx) {
            // getVocab(withAddedTokens: bool = true)
            let with_added_tokens = cx.extract_opt::<bool>(0)?.unwrap_or(true);

            let this = cx.this();
            let guard = cx.lock();
            let vocab = this.borrow(&guard)
                .tokenizer.read().unwrap()
                .get_vocab(with_added_tokens);

            let js_vocab = JsObject::new(&mut cx);
            for (token, id) in vocab {
                let js_token = cx.string(token);
                let js_id = cx.number(id as f64);
                js_vocab.set(&mut cx, js_token, js_id)?;
            }

            Ok(js_vocab.upcast())
        }

        method getVocabSize(mut cx) {
            // getVocabSize(withAddedTokens: bool = true)
            let with_added_tokens = cx.extract_opt::<bool>(0)?.unwrap_or(true);

            let this = cx.this();
            let guard = cx.lock();
            let size = this.borrow(&guard)
                .tokenizer.read().unwrap()
                .get_vocab_size(with_added_tokens);

            Ok(cx.number(size as f64).upcast())
        }

        method normalize(mut cx) {
            // normalize(sentence: String) -> String
            let sentence = cx.extract::<String>(0)?;

            let this = cx.this();
            let guard = cx.lock();

            let normalized = this.borrow(&guard)
                .tokenizer.read().unwrap()
                .normalize(&sentence)
                .map(|s| s.get().to_owned())
                .map_err(|e| Error(format!("{}", e)))?;

            Ok(cx.string(normalized).upcast())
        }

        method encode(mut cx) {
            // type InputSequence = string | string[];
            // encode(
            //   sentence: InputSequence,
            //   pair?: InputSequence,
            //   options?: {
            //     addSpecialTokens?: boolean,
            //     isPretokenized?: boolean,
            //   } | (err, encoding) -> void,
            //   __callback: (err, encoding) -> void
            // )

            // Start by extracting options and callback
            let (options, callback) = match cx.extract_opt::<EncodeOptions>(2) {
                // Options were there, and extracted
                Ok(Some(options)) => {
                    (options, cx.argument::<JsFunction>(3)?)
                },
                // Options were undefined or null
                Ok(None) => {
                    (EncodeOptions::default(), cx.argument::<JsFunction>(3)?)
                }
                // Options not specified, callback instead
                Err(_) => {
                    (EncodeOptions::default(), cx.argument::<JsFunction>(2)?)
                }
            };

            // Then we extract our input sequences
            let sentence: tk::InputSequence = if options.is_pretokenized {
                cx.extract::<PreTokenizedInputSequence>(0)
                    .map_err(|_| Error("encode with isPretokenized=true expect string[]".into()))?
                    .into()
            } else {
                cx.extract::<TextInputSequence>(0)
                    .map_err(|_| Error("encode with isPreTokenized=false expect string".into()))?
                    .into()
            };
            let pair: Option<tk::InputSequence> = if options.is_pretokenized {
                cx.extract_opt::<PreTokenizedInputSequence>(1)
                    .map_err(|_| Error("encode with isPretokenized=true expect string[]".into()))?
                    .map(|v| v.into())
            } else {
                cx.extract_opt::<TextInputSequence>(1)
                    .map_err(|_| Error("encode with isPreTokenized=false expect string".into()))?
                    .map(|v| v.into())
            };
            let input: tk::EncodeInput = match pair {
                Some(pair) => (sentence, pair).into(),
                None => sentence.into()
            };

            let this = cx.this();
            let guard = cx.lock();

            let task = EncodeTask::Single(
                this.borrow(&guard).clone(), Some(input), options.add_special_tokens
            );
            task.schedule(callback);

            Ok(cx.undefined().upcast())
        }

        method encodeBatch(mut cx) {
            // type InputSequence = string | string[];
            // type EncodeInput = (InputSequence | [InputSequence, InputSequence])[]
            // encode_batch(
            //   inputs: EncodeInput[],
            //   options?: {
            //     addSpecialTokens?: boolean,
            //     isPretokenized?: boolean,
            //   } | (err, encodings) -> void,
            //   __callback: (err, encodings) -> void
            // )

            // Start by extracting options and callback
            let (options, callback) = match cx.extract_opt::<EncodeOptions>(1) {
                // Options were there, and extracted
                Ok(Some(options)) => {
                    (options, cx.argument::<JsFunction>(2)?)
                },
                // Options were undefined or null
                Ok(None) => {
                    (EncodeOptions::default(), cx.argument::<JsFunction>(2)?)
                }
                // Options not specified, callback instead
                Err(_) => {
                    (EncodeOptions::default(), cx.argument::<JsFunction>(1)?)
                }
            };

            let inputs: Vec<tk::EncodeInput> = if options.is_pretokenized {
                cx.extract_vec::<PreTokenizedEncodeInput>(0)
                    .map_err(|_| Error(
                        "encodeBatch with isPretokenized=true expects input to be `EncodeInput[]` \
                        with `EncodeInput = string[] | [string[], string[]]`".into()))?
                    .into_iter().map(|v| v.into()).collect()
            } else {
                cx.extract_vec::<TextEncodeInput>(0)
                    .map_err(|_| Error(
                        "encodeBatch with isPretokenized=false expects input to be `EncodeInput[]` \
                        with `EncodeInput = string | [string, string]`".into()))?
                    .into_iter().map(|v| v.into()).collect()
            };

            let this = cx.this();
            let guard = cx.lock();

            let task = EncodeTask::Batch(
                this.borrow(&guard).clone(), Some(inputs), options.add_special_tokens
            );
            task.schedule(callback);

            Ok(cx.undefined().upcast())
        }

        method decode(mut cx) {
            // decode(ids: number[], skipSpecialTokens: bool, callback)

            let ids = cx.extract_vec::<u32>(0)?;
            let skip_special_tokens = cx.extract::<bool>(1)?;
            let callback = cx.argument::<JsFunction>(2)?;

            let this = cx.this();
            let guard = cx.lock();

            let task = DecodeTask::Single(
                this.borrow(&guard).clone(), ids, skip_special_tokens
            );
            task.schedule(callback);

            Ok(cx.undefined().upcast())
        }

        method decodeBatch(mut cx) {
            // decodeBatch(sequences: number[][], skipSpecialTokens: bool, callback)

            let sentences = cx.extract_vec::<Vec<u32>>(0)?;
            let skip_special_tokens = cx.extract::<bool>(1)?;
            let callback = cx.argument::<JsFunction>(2)?;

            let this = cx.this();
            let guard = cx.lock();

            let task = DecodeTask::Batch(
                this.borrow(&guard).clone(), sentences, skip_special_tokens
            );
            task.schedule(callback);

            Ok(cx.undefined().upcast())
        }

        method tokenToId(mut cx) {
            // tokenToId(token: string): number | undefined

            let token = cx.extract::<String>(0)?;

            let this = cx.this();
            let guard = cx.lock();
            let id = this.borrow(&guard)
                .tokenizer.read().unwrap()
                .token_to_id(&token);

            if let Some(id) = id {
                Ok(cx.number(id).upcast())
            } else {
                Ok(cx.undefined().upcast())
            }
        }

        method idToToken(mut cx) {
            // idToToken(id: number): string | undefined

            let id = cx.extract::<u32>(0)?;

            let this = cx.this();
            let guard = cx.lock();
            let token = this.borrow(&guard)
                .tokenizer.read().unwrap()
                .id_to_token(id).map(|t| t.to_owned());

            if let Some(token) = token {
                Ok(cx.string(token).upcast())
            } else {
                Ok(cx.undefined().upcast())
            }
        }

        method addTokens(mut cx) {
            // addTokens(tokens: (string | AddedToken)[]): number

            let tokens = cx.extract_vec::<AddedToken>(0)?
                .into_iter()
                .map(|token| token.into())
                .collect::<Vec<_>>();

            let mut this = cx.this();
            let guard = cx.lock();
            let added = this.borrow_mut(&guard)
                .tokenizer.write().unwrap()
                .add_tokens(&tokens);

            Ok(cx.number(added as f64).upcast())
        }

        method addSpecialTokens(mut cx) {
            // addSpecialTokens(tokens: (string | AddedToken)[]): number

            let tokens = cx.extract_vec::<SpecialToken>(0)?
                .into_iter()
                .map(|token| token.0)
                .collect::<Vec<_>>();

            let mut this = cx.this();
            let guard = cx.lock();
            let added = this.borrow_mut(&guard)
                .tokenizer.write().unwrap()
                .add_special_tokens(&tokens);

            Ok(cx.number(added as f64).upcast())
        }

        method setTruncation(mut cx) {
            // setTruncation(
            //   maxLength: number,
            //   options?: { stride?: number; strategy?: string }
            // )

            let max_length = cx.extract::<usize>(0)?;
            let mut options = cx.extract_opt::<TruncationParams>(1)?
                .map_or_else(tk::TruncationParams::default, |p| p.0);
            options.max_length = max_length;

            let params_obj = neon_serde::to_value(&mut cx, &TruncationParams(options.clone()))?;
            let mut this = cx.this();
            let guard = cx.lock();
            this.borrow_mut(&guard)
                .tokenizer.write().unwrap()
                .with_truncation(Some(options));

            Ok(params_obj)
        }

        method disableTruncation(mut cx) {
            // disableTruncation()

            let mut this = cx.this();
            let guard = cx.lock();
            this.borrow_mut(&guard)
                .tokenizer.write().unwrap()
                .with_truncation(None);

            Ok(cx.undefined().upcast())
        }

        method setPadding(mut cx) {
            // setPadding(options?: {
            //   direction?: "left" | "right",
            //   padId?: number,
            //   padTypeId?: number,
            //   padToken?: string,
            //   maxLength?: number
            //  })

            let options = cx.extract_opt::<PaddingParams>(0)?
                .map_or_else(tk::PaddingParams::default, |p| p.0);

            let params_obj = neon_serde::to_value(&mut cx, &PaddingParams(options.clone()))?;
            let mut this = cx.this();
            let guard = cx.lock();
            this.borrow_mut(&guard)
                .tokenizer.write().unwrap()
                .with_padding(Some(options));

            Ok(params_obj)
        }

        method disablePadding(mut cx) {
            // disablePadding()

            let mut this = cx.this();
            let guard = cx.lock();
            this.borrow_mut(&guard)
                .tokenizer.write().unwrap()
                .with_padding(None);

            Ok(cx.undefined().upcast())
        }

        method train(mut cx) {
            // train(trainer: JsTrainer, files: string[])

            let trainer = cx.argument::<JsTrainer>(0)?;
            let files = cx.extract::<Vec<String>>(1)?;

            let mut this = cx.this();
            let guard = cx.lock();

            let trainer = trainer.borrow(&guard).clone();
            let new_tokenizer = this.borrow_mut(&guard)
                .tokenizer.write().unwrap()
                .train(&trainer, files)
                .map_err(|e| Error(format!("{}", e)))?;

            Ok(cx.undefined().upcast())
        }

        method postProcess(mut cx) {
            // postProcess(
            //   encoding: Encoding,
            //   pair?: Encoding,
            //   addSpecialTokens: boolean = true
            // ): Encoding

            let encoding = cx.extract::<Encoding>(0)?;
            let pair = cx.extract_opt::<Encoding>(1)?;
            let add_special_tokens = cx.extract_opt::<bool>(2)?.unwrap_or(true);

            let this = cx.this();
            let guard = cx.lock();
            let encoding = this.borrow(&guard)
                .tokenizer.read().unwrap()
                .post_process(encoding.into(), pair.map(|p| p.into()), add_special_tokens)
                .map_err(|e| Error(format!("{}", e)))?;

            let mut js_encoding = JsEncoding::new::<_, JsEncoding, _>(&mut cx, vec![])?;
            let guard = cx.lock();
            js_encoding
                .borrow_mut(&guard)
                .encoding
                .make_owned(Box::new(encoding));

            Ok(js_encoding.upcast())
        }

        method getModel(mut cx) {
            // getModel(): Model

            let this = cx.this();
            let guard = cx.lock();
            let model = this.borrow(&guard)
                .tokenizer.read().unwrap()
                .get_model()
                .model
                .clone();

            let mut js_model = JsModel::new::<_, JsModel, _>(&mut cx, vec![])?;
            let guard = cx.lock();
            js_model.borrow_mut(&guard).model = model;

            Ok(js_model.upcast())
        }

        method setModel(mut cx) {
            // setModel(model: JsModel)

            let model = cx.argument::<JsModel>(0)?;
            let mut this = cx.this();
            let guard = cx.lock();

            let model = model.borrow(&guard).clone();
            this.borrow_mut(&guard)
                .tokenizer.write().unwrap()
                .with_model(model);

            Ok(cx.undefined().upcast())
        }

        method getNormalizer(mut cx) {
            // getNormalizer(): Normalizer | undefined

            let this = cx.this();
            let guard = cx.lock();
            let normalizer = this.borrow(&guard)
                .tokenizer.read().unwrap()
                .get_normalizer().cloned();

            if let Some(normalizer) = normalizer {
                let mut js_normalizer = JsNormalizer::new::<_, JsNormalizer, _>(&mut cx, vec![])?;
                let guard = cx.lock();

                js_normalizer.borrow_mut(&guard).normalizer = normalizer.normalizer;
                Ok(js_normalizer.upcast())
            } else {
                Ok(cx.undefined().upcast())
            }
        }

        method setNormalizer(mut cx) {
            // setNormalizer(normalizer: Normalizer)

            let normalizer = cx.argument::<JsNormalizer>(0)?;
            let mut this = cx.this();
            let guard = cx.lock();

            let normalizer = normalizer.borrow(&guard).clone();
            this.borrow_mut(&guard)
                .tokenizer.write().unwrap()
                .with_normalizer(normalizer);

            Ok(cx.undefined().upcast())
        }

        method getPreTokenizer(mut cx) {
            // getPreTokenizer(): PreTokenizer | undefined

            let this = cx.this();
            let guard = cx.lock();
            let pretok = this.borrow(&guard)
                .tokenizer.read().unwrap()
                .get_pre_tokenizer().cloned();

            if let Some(pretok) = pretok {
                let mut js_pretok = JsPreTokenizer::new::<_, JsPreTokenizer, _>(&mut cx, vec![])?;
                let guard = cx.lock();

                js_pretok.borrow_mut(&guard).pretok = pretok.pretok;
                Ok(js_pretok.upcast())
            } else {
                Ok(cx.undefined().upcast())
            }
        }

        method setPreTokenizer(mut cx) {
            // setPreTokenizer(pretokenizer: PreTokenizer)

            let pretok = cx.argument::<JsPreTokenizer>(0)?;
            let mut this = cx.this();
            let guard = cx.lock();

            let pretok = pretok.borrow(&guard).clone();
            this.borrow_mut(&guard)
                .tokenizer.write().unwrap()
                .with_pre_tokenizer(pretok);

            Ok(cx.undefined().upcast())
        }

        method getPostProcessor(mut cx) {
            // getPostProcessor(): PostProcessor | undefined

            let this = cx.this();
            let guard = cx.lock();
            let processor = this.borrow(&guard)
                .tokenizer.read().unwrap()
                .get_post_processor().cloned();

            if let Some(processor) = processor {
                let mut js_processor =
                    JsPostProcessor::new::<_, JsPostProcessor, _>(&mut cx, vec![])?;
                let guard = cx.lock();

                js_processor.borrow_mut(&guard).processor = processor.processor;
                Ok(js_processor.upcast())
            } else {
                Ok(cx.undefined().upcast())
            }
        }

        method setPostProcessor(mut cx) {
            // setPostProcessor(processor: PostProcessor)

            let processor = cx.argument::<JsPostProcessor>(0)?;
            let mut this = cx.this();
            let guard = cx.lock();

            let processor = processor.borrow(&guard).clone();
            this.borrow_mut(&guard)
                .tokenizer.write().unwrap()
                .with_post_processor(processor);

            Ok(cx.undefined().upcast())
        }

        method getDecoder(mut cx) {
            // getDecoder(): Decoder | undefined

            let this = cx.this();
            let guard = cx.lock();
            let decoder = this.borrow(&guard)
                .tokenizer.read().unwrap()
                .get_decoder().cloned();

            if let Some(decoder) = decoder {
                let mut js_decoder = JsDecoder::new::<_, JsDecoder, _>(&mut cx, vec![])?;
                let guard = cx.lock();

                js_decoder.borrow_mut(&guard).decoder = decoder.decoder;
                Ok(js_decoder.upcast())
            } else {
                Ok(cx.undefined().upcast())
            }
        }

        method setDecoder(mut cx) {
            // setDecoder(decoder: Decoder)

            let decoder = cx.argument::<JsDecoder>(0)?;
            let mut this = cx.this();
            let guard = cx.lock();

            let decoder = decoder.borrow(&guard).clone();
            this.borrow_mut(&guard)
                .tokenizer.write().unwrap()
                .with_decoder(decoder);

            Ok(cx.undefined().upcast())
        }
    }
}

pub fn tokenizer_from_string(mut cx: FunctionContext) -> JsResult<JsTokenizer> {
    let s = cx.extract::<String>(0)?;

    let tokenizer: tk::tokenizer::TokenizerImpl<
        Model,
        Normalizer,
        PreTokenizer,
        Processor,
        Decoder,
    > = s.parse().map_err(|e| Error(format!("{}", e)))?;

    let mut js_tokenizer = JsTokenizer::new::<_, JsTokenizer, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    js_tokenizer.borrow_mut(&guard).tokenizer = Arc::new(RwLock::new(tokenizer));

    Ok(js_tokenizer)
}

pub fn tokenizer_from_file(mut cx: FunctionContext) -> JsResult<JsTokenizer> {
    let s = cx.extract::<String>(0)?;

    let tokenizer =
        tk::tokenizer::TokenizerImpl::from_file(s).map_err(|e| Error(format!("{}", e)))?;

    let mut js_tokenizer = JsTokenizer::new::<_, JsTokenizer, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    js_tokenizer.borrow_mut(&guard).tokenizer = Arc::new(RwLock::new(tokenizer));

    Ok(js_tokenizer)
}

pub fn register(m: &mut ModuleContext, prefix: &str) -> Result<(), neon::result::Throw> {
    m.export_class::<JsAddedToken>(&format!("{}_AddedToken", prefix))?;
    m.export_class::<JsTokenizer>(&format!("{}_Tokenizer", prefix))?;
    m.export_function(
        &format!("{}_Tokenizer_from_string", prefix),
        tokenizer_from_string,
    )?;
    m.export_function(
        &format!("{}_Tokenizer_from_file", prefix),
        tokenizer_from_file,
    )?;
    Ok(())
}
