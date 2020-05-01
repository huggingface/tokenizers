extern crate tokenizers as tk;

use crate::container::Container;
use crate::decoders::JsDecoder;
use crate::encoding::JsEncoding;
use crate::models::JsModel;
use crate::normalizers::JsNormalizer;
use crate::pre_tokenizers::JsPreTokenizer;
use crate::processors::JsPostProcessor;
use crate::tasks::tokenizer::{DecodeTask, EncodeTask, WorkingTokenizer};
use crate::trainers::JsTrainer;
use neon::prelude::*;
use serde::de::DeserializeOwned;

use tk::tokenizer::{
    PaddingDirection, PaddingParams, PaddingStrategy, TruncationParams, TruncationStrategy,
};

pub struct AddedToken {
    pub token: tk::tokenizer::AddedToken,
}

declare_types! {
    pub class JsAddedToken for AddedToken {
        init(mut cx) {
            // init(content: string,
            //    options?: { singleWord?: boolean = False, leftStrip?: boolean = False, rightStrip?: boolean = False }
            // )

            let mut token = tk::tokenizer::AddedToken::from(cx.argument::<JsString>(0)?.value());

            let options = cx.argument_opt(1);
            if let Some(options) = options {
                if let Ok(options) = options.downcast::<JsObject>() {
                    if let Ok(single_word) = options.get(&mut cx, "singleWord") {
                        if single_word.downcast::<JsUndefined>().is_err() {
                            token = token.single_word(single_word.downcast::<JsBoolean>().or_throw(&mut cx)?.value());
                        }
                    }

                    if let Ok(left_strip) = options.get(&mut cx, "leftStrip") {
                        if left_strip.downcast::<JsUndefined>().is_err() {
                            token = token.lstrip(left_strip.downcast::<JsBoolean>().or_throw(&mut cx)?.value());
                        }
                    }

                    if let Ok(right_strip) = options.get(&mut cx, "rightStrip") {
                        if right_strip.downcast::<JsUndefined>().is_err() {
                            token = token.rstrip(right_strip.downcast::<JsBoolean>().or_throw(&mut cx)?.value());
                        }
                    }
                }
            }

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

pub struct Error(String);
impl<T> From<T> for Error
where
    T: std::fmt::Display,
{
    fn from(e: T) -> Self {
        Self(format!("{}", e))
    }
}
impl From<Error> for neon::result::Throw {
    fn from(err: Error) -> Self {
        let msg = err.0;
        unsafe {
            neon_runtime::error::throw_error_from_utf8(msg.as_ptr(), msg.len() as i32);
            neon::result::Throw
        }
    }
}

pub type LibResult<T> = std::result::Result<T, Error>;

trait FromJsValue: Sized {
    fn from_value<'c, C: Context<'c>>(from: Handle<'c, JsValue>, cx: &mut C) -> LibResult<Self>;
}

impl<T> FromJsValue for T
where
    T: DeserializeOwned,
{
    fn from_value<'c, C: Context<'c>>(from: Handle<'c, JsValue>, cx: &mut C) -> LibResult<Self> {
        let val: T = neon_serde::from_value(cx, from)?;
        Ok(val)
    }
}

trait Extract {
    fn extract<T: FromJsValue>(&mut self, pos: i32) -> LibResult<T>;
    fn extract_opt<T: FromJsValue>(&mut self, pos: i32) -> LibResult<Option<T>>;
    fn extract_vec<T: FromJsValue>(&mut self, pos: i32) -> LibResult<Vec<T>>;
    fn extract_vec_opt<T: FromJsValue>(&mut self, pos: i32) -> LibResult<Option<Vec<T>>>;
}

impl<'c, T: neon::object::This> Extract for CallContext<'c, T> {
    fn extract<E: FromJsValue>(&mut self, pos: i32) -> LibResult<E> {
        let val = self
            .argument_opt(pos)
            .ok_or_else(|| Error(format!("Argument {} is missing", pos)))?;
        let ext = E::from_value(val, self)?;
        Ok(ext)
    }

    fn extract_opt<E: FromJsValue>(&mut self, pos: i32) -> LibResult<Option<E>> {
        let val = self.argument_opt(pos);
        match val {
            None => Ok(None),
            Some(v) => {
                // For any optional value, we accept both `undefined` and `null`
                if v.downcast::<JsNull>().is_ok() || v.downcast::<JsUndefined>().is_ok() {
                    Ok(None)
                } else {
                    Ok(Some(E::from_value(v, self)?))
                }
            }
        }
    }

    fn extract_vec<E: FromJsValue>(&mut self, pos: i32) -> LibResult<Vec<E>> {
        let vec = self
            .argument_opt(pos)
            .ok_or_else(|| Error(format!("Argument {} is missing", pos)))?
            .downcast::<JsArray>()?
            .to_vec(self)?;

        vec.into_iter().map(|v| E::from_value(v, self)).collect()
    }

    fn extract_vec_opt<E: FromJsValue>(&mut self, pos: i32) -> LibResult<Option<Vec<E>>> {
        self.argument_opt(pos)
            .map(|v| {
                let vec = v.downcast::<JsArray>()?.to_vec(self)?;
                Ok(vec
                    .into_iter()
                    .map(|v| E::from_value(v, self))
                    .collect::<LibResult<Vec<_>>>()?)
            })
            .map_or(Ok(None), |v| v.map(Some))
    }
}

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

#[allow(non_snake_case)]
#[derive(Debug, Serialize, Deserialize)]
struct EncodeOptions {
    #[serde(default)]
    isPretokenized: bool,
    #[serde(default)]
    addSpecialTokens: bool,
}
impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            isPretokenized: false,
            addSpecialTokens: true,
        }
    }
}

/// Tokenizer
pub struct Tokenizer {
    tokenizer: tk::tokenizer::Tokenizer,

    /// Whether we have a running task. We keep this to make sure we never
    /// modify the underlying tokenizer while a task is running
    running_task: std::sync::Arc<()>,
}

impl Tokenizer {
    pub fn prepare_for_task(&self) -> WorkingTokenizer {
        unsafe { WorkingTokenizer::new(&self.tokenizer, self.running_task.clone()) }
    }
}

declare_types! {
    pub class JsTokenizer for Tokenizer {
        init(mut cx) {
            // init(model: JsModel)
            let mut model = cx.argument::<JsModel>(0)?;
            if let Some(instance) = {
                let guard = cx.lock();
                let mut model = model.borrow_mut(&guard);
                model.model.to_pointer()
            } {
                Ok(Tokenizer {
                    tokenizer: tk::tokenizer::Tokenizer::new(instance),
                    running_task: std::sync::Arc::new(())
                })
            } else {
                cx.throw_error("The Model is already being used in another Tokenizer")
            }
        }

        method runningTasks(mut cx) {
            // runningTasks(): number
            let running = {
                let this = cx.this();
                let guard = cx.lock();
                let count = std::sync::Arc::strong_count(&this.borrow(&guard).running_task);
                if count > 0 { count - 1 } else { 0 }
            };
            Ok(cx.number(running as f64).upcast())
        }

        method getVocab(mut cx) {
            // getVocab(withAddedTokens: bool = true)
            let mut with_added_tokens = true;
            if let Some(arg) = cx.argument_opt(0) {
                if arg.downcast::<JsUndefined>().is_err() {
                    with_added_tokens = arg.downcast::<JsBoolean>()
                        .or_throw(&mut cx)?
                        .value() as bool;
                }
            }

            let this = cx.this();
            let guard = cx.lock();
            let vocab = this.borrow(&guard)
                .tokenizer
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
            let mut with_added_tokens = true;
            if let Some(arg) = cx.argument_opt(0) {
                if arg.downcast::<JsUndefined>().is_err() {
                    with_added_tokens = arg.downcast::<JsBoolean>()
                        .or_throw(&mut cx)?
                        .value() as bool;
                }
            }

            let this = cx.this();
            let guard = cx.lock();
            let size = this.borrow(&guard)
                .tokenizer
                .get_vocab_size(with_added_tokens);

            Ok(cx.number(size as f64).upcast())
        }

        method normalize(mut cx) {
            // normalize(sentence: String) -> String
            let sentence = cx.argument::<JsString>(0)?.value();

            let this = cx.this();
            let guard = cx.lock();

            let result = {
                this.borrow(&guard)
                    .tokenizer
                    .normalize(&sentence)
                    .map(|s| s.get().to_owned())
            };
            let normalized = result
                .map_err(|e| {
                    cx.throw_error::<_, ()>(format!("{}", e))
                        .unwrap_err()
                })?;

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
            let sentence: tk::InputSequence = if options.isPretokenized {
                cx.extract::<PreTokenizedInputSequence>(0)?.into()
            } else {
                cx.extract::<TextInputSequence>(0)?.into()
            };
            let pair: Option<tk::InputSequence> = if options.isPretokenized {
                cx.extract_opt::<PreTokenizedInputSequence>(1)?.map(|v| v.into())
            } else {
                cx.extract_opt::<TextInputSequence>(1)?.map(|v| v.into())
            };
            let input: tk::EncodeInput = match pair {
                Some(pair) => (sentence, pair).into(),
                None => sentence.into()
            };

            let worker = {
                let this = cx.this();
                let guard = cx.lock();
                let worker = this.borrow(&guard).prepare_for_task();
                worker
            };

            let task = EncodeTask::Single(worker, Some(input), options.addSpecialTokens);
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

            let inputs: Vec<tk::EncodeInput> = if options.isPretokenized {
                cx.extract_vec::<PreTokenizedEncodeInput>(0)?
                    .into_iter().map(|v| v.into()).collect()
            } else {
                cx.extract_vec::<TextEncodeInput>(0)?
                    .into_iter().map(|v| v.into()).collect()
            };

            let worker = {
                let this = cx.this();
                let guard = cx.lock();
                let worker = this.borrow(&guard).prepare_for_task();
                worker
            };

            let task = EncodeTask::Batch(worker, Some(inputs), options.addSpecialTokens);
            task.schedule(callback);
            Ok(cx.undefined().upcast())
        }

        method decode(mut cx) {
            // decode(ids: number[], skipSpecialTokens: bool, callback)

            let ids = cx.argument::<JsArray>(0)?.to_vec(&mut cx)?
                .into_iter()
                .map(|id| {
                    id.downcast::<JsNumber>()
                        .or_throw(&mut cx)
                        .map(|v| v.value() as u32)
                })
                .collect::<NeonResult<Vec<_>>>()?;
            let skip_special_tokens = cx.argument::<JsBoolean>(1)?.value();
            let callback = cx.argument::<JsFunction>(2)?;

            let worker = {
                let this = cx.this();
                let guard = cx.lock();
                let worker = this.borrow(&guard).prepare_for_task();
                worker
            };

            let task = DecodeTask::Single(worker, ids, skip_special_tokens);
            task.schedule(callback);
            Ok(cx.undefined().upcast())
        }

        method decodeBatch(mut cx) {
            // decodeBatch(sequences: number[][], skipSpecialTokens: bool, callback)

            let sentences = cx.argument::<JsArray>(0)?
                .to_vec(&mut cx)?
                .into_iter()
                .map(|sentence| {
                    sentence.downcast::<JsArray>()
                        .or_throw(&mut cx)?
                        .to_vec(&mut cx)?
                        .into_iter()
                        .map(|id| {
                            id.downcast::<JsNumber>()
                                .or_throw(&mut cx)
                                .map(|v| v.value() as u32)
                        })
                        .collect::<NeonResult<Vec<_>>>()
                }).collect::<NeonResult<Vec<_>>>()?;

            let skip_special_tokens = cx.argument::<JsBoolean>(1)?.value();
            let callback = cx.argument::<JsFunction>(2)?;

            let worker = {
                let this = cx.this();
                let guard = cx.lock();
                let worker = this.borrow(&guard).prepare_for_task();
                worker
            };

            let task = DecodeTask::Batch(worker, sentences, skip_special_tokens);
            task.schedule(callback);
            Ok(cx.undefined().upcast())
        }

        method tokenToId(mut cx) {
            // tokenToId(token: string): number | undefined

            let token = cx.argument::<JsString>(0)?.value();

            let this = cx.this();
            let guard = cx.lock();
            let id = this.borrow(&guard).tokenizer.token_to_id(&token);

            if let Some(id) = id {
                Ok(cx.number(id).upcast())
            } else {
                Ok(cx.undefined().upcast())
            }
        }

        method idToToken(mut cx) {
            // idToToken(id: number): string | undefined

            let id = cx.argument::<JsNumber>(0)?.value() as u32;

            let this = cx.this();
            let guard = cx.lock();
            let token = this.borrow(&guard).tokenizer.id_to_token(id);

            if let Some(token) = token {
                Ok(cx.string(token).upcast())
            } else {
                Ok(cx.undefined().upcast())
            }
        }

        method addTokens(mut cx) {
            // addTokens(tokens: (string | AddedToken)[]): number

            let tokens = cx.argument::<JsArray>(0)?
                .to_vec(&mut cx)?
                .into_iter()
                .map(|token| {
                    if let Ok(token) = token.downcast::<JsString>() {
                        Ok(tk::tokenizer::AddedToken::from(token.value()))
                    } else if let Ok(token) = token.downcast::<JsAddedToken>() {
                        let guard = cx.lock();
                        let token = token.borrow(&guard);
                        Ok(token.token.clone())
                    } else {
                        cx.throw_error("Input must be `(string | AddedToken)[]`")
                    }
                })
                .collect::<NeonResult<Vec<_>>>()?;

            let mut this = cx.this();
            let guard = cx.lock();
            let added = this.borrow_mut(&guard).tokenizer.add_tokens(&tokens);

            Ok(cx.number(added as f64).upcast())
        }

        method addSpecialTokens(mut cx) {
            // addSpecialTokens(tokens: (string | AddedToken)[]): number

            let tokens = cx.argument::<JsArray>(0)?
                .to_vec(&mut cx)?
                .into_iter()
                .map(|token| {
                    if let Ok(token) = token.downcast::<JsString>() {
                        Ok(tk::tokenizer::AddedToken::from(token.value()))
                    } else if let Ok(token) = token.downcast::<JsAddedToken>() {
                        let guard = cx.lock();
                        let token = token.borrow(&guard);
                        Ok(token.token.clone())
                    } else {
                        cx.throw_error("Input must be `(string | AddedToken)[]`")
                    }
                })
                .collect::<NeonResult<Vec<_>>>()?;

            let mut this = cx.this();
            let guard = cx.lock();
            let added = this.borrow_mut(&guard)
                .tokenizer
                .add_special_tokens(&tokens);

            Ok(cx.number(added as f64).upcast())
        }

        method setTruncation(mut cx) {
            // setTruncation(maxLength: number, options?: { stride?: number; strategy?: string })
            let max_length = cx.argument::<JsNumber>(0)?.value() as usize;

            let mut stride = 0;
            let mut strategy = TruncationStrategy::LongestFirst;

            let options = cx.argument_opt(1);
            if let Some(options) = options {
                if let Ok(options) = options.downcast::<JsObject>() {
                    if let Ok(stride_opt) = options.get(&mut cx, "stride") {
                        if stride_opt.downcast::<JsUndefined>().is_err() {
                            stride = stride_opt.downcast::<JsNumber>().or_throw(&mut cx)?.value() as usize;
                        }
                    }
                    if let Ok(strat_opt) = options.get(&mut cx, "strategy") {
                        if strat_opt.downcast::<JsUndefined>().is_err() {
                            let strat_opt = strat_opt.downcast::<JsString>().or_throw(&mut cx)?.value();
                            match &strat_opt[..] {
                                "longest_first" => strategy = TruncationStrategy::LongestFirst,
                                "only_first" => strategy = TruncationStrategy::OnlyFirst,
                                "only_second" => strategy = TruncationStrategy::OnlySecond,
                                _ => return cx.throw_error("strategy can only be 'longest_first', 'only_first' or 'only_second'"),
                            }
                        }
                    }
                }
            }

            let mut this = cx.this();
            {
                let guard = cx.lock();
                let mut tokenizer = this.borrow_mut(&guard);
                tokenizer.tokenizer.with_truncation(Some(TruncationParams {
                    max_length,
                    stride,
                    strategy,
                }));
            }

            let params_object = JsObject::new(&mut cx);
            let obj_length = cx.number(max_length as f64);
            let obj_stride = cx.number(stride as f64);
            let obj_strat = cx.string(strategy);

            params_object.set(&mut cx, "maxLength", obj_length).unwrap();
            params_object.set(&mut cx, "stride", obj_stride).unwrap();
            params_object.set(&mut cx, "strategy", obj_strat).unwrap();

            Ok(params_object.upcast())
        }

        method disableTruncation(mut cx) {
            let mut this = cx.this();
            let guard = cx.lock();
            this.borrow_mut(&guard).tokenizer.with_truncation(None);
            Ok(cx.undefined().upcast())
        }

        method setPadding(mut cx) {
            // setPadding(options?: { direction?: "left" | "right"; padId?: number?; padTypeId?: number?; padToken: string; maxLength?: number })
            let mut direction = PaddingDirection::Right;
            let mut pad_id: u32 = 0;
            let mut pad_type_id: u32 = 0;
            let mut pad_token = String::from("[PAD]");
            let mut max_length: Option<usize> = None;

            let options = cx.argument_opt(0);
            if let Some(options) = options {
                if let Ok(options) = options.downcast::<JsObject>() {
                    if let Ok(dir) = options.get(&mut cx, "direction") {
                        if dir.downcast::<JsUndefined>().is_err() {
                            let dir = dir.downcast::<JsString>().or_throw(&mut cx)?.value();
                            match &dir[..] {
                                "left" => direction = PaddingDirection::Left,
                                "right" => direction = PaddingDirection::Right,
                                _ => return cx.throw_error("direction can only be 'left' or 'right'"),
                            }
                        }
                    }
                    if let Ok(p_id) = options.get(&mut cx, "padId") {
                        if p_id.downcast::<JsUndefined>().is_err() {
                            pad_id = p_id.downcast::<JsNumber>().or_throw(&mut cx)?.value() as u32;
                        }
                    }
                    if let Ok(p_type_id) = options.get(&mut cx, "padTypeId") {
                        if p_type_id.downcast::<JsUndefined>().is_err() {
                            pad_type_id = p_type_id.downcast::<JsNumber>().or_throw(&mut cx)?.value() as u32;
                        }
                    }
                    if let Ok(p_token) = options.get(&mut cx, "padToken") {
                        if p_token.downcast::<JsUndefined>().is_err() {
                            pad_token = p_token.downcast::<JsString>().or_throw(&mut cx)?.value();
                        }
                    }
                    if let Ok(max_l) = options.get(&mut cx, "maxLength") {
                        if max_l.downcast::<JsUndefined>().is_err() {
                            max_length = Some(max_l.downcast::<JsNumber>().or_throw(&mut cx)?.value() as usize);
                        }
                    }
                }
            }

            let strategy = if let Some(max_length) = max_length {
                PaddingStrategy::Fixed(max_length)
            } else {
                PaddingStrategy::BatchLongest
            };

            let mut this = cx.this();
            {
                let guard = cx.lock();
                let mut tokenizer = this.borrow_mut(&guard);
                tokenizer.tokenizer.with_padding(Some(PaddingParams {
                    strategy,
                    direction,
                    pad_id,
                    pad_type_id,
                    pad_token: pad_token.to_owned(),
                }));
            }

            let params_object = JsObject::new(&mut cx);
            if let Some(max_length) = max_length {
                let obj_length = cx.number(max_length as f64);
                params_object.set(&mut cx, "maxLength", obj_length).unwrap();
            }
            let obj_pad_id = cx.number(pad_id);
            let obj_pad_type_id = cx.number(pad_type_id);
            let obj_pad_token = cx.string(pad_token);
            let obj_direction = cx.string(direction);
            params_object.set(&mut cx, "padId", obj_pad_id).unwrap();
            params_object.set(&mut cx, "padTypeId", obj_pad_type_id).unwrap();
            params_object.set(&mut cx, "padToken", obj_pad_token).unwrap();
            params_object.set(&mut cx, "direction", obj_direction).unwrap();

            Ok(params_object.upcast())
        }

        method disablePadding(mut cx) {
            let mut this = cx.this();
            let guard = cx.lock();
            this.borrow_mut(&guard).tokenizer.with_padding(None);
            Ok(cx.undefined().upcast())
        }

        method train(mut cx) {
            // train(trainer: JsTrainer, files: string[])

            let trainer = cx.argument::<JsTrainer>(0)?;
            let files = cx.argument::<JsArray>(1)?.to_vec(&mut cx)?.into_iter().map(|file| {
                Ok(file.downcast::<JsString>().or_throw(&mut cx)?.value())
            }).collect::<NeonResult<Vec<_>>>()?;

            let mut this = cx.this();
            let guard = cx.lock();
            let res = trainer.borrow(&guard).trainer.execute(|trainer| {
                let res = this.borrow_mut(&guard).tokenizer.train(trainer.unwrap(), files);
                res
            });
            res.map_err(|e| cx.throw_error::<_, ()>(format!("{}", e)).unwrap_err())?;

            Ok(cx.undefined().upcast())
        }

        method postProcess(mut cx) {
            // postProcess(
            //   encoding: Encoding,
            //   pair?: Encoding,
            //   addSpecialTokens: boolean = true
            // ): Encoding

            let encoding = {
                let encoding = cx.argument::<JsEncoding>(0)?;
                let guard = cx.lock();
                let encoding = encoding
                    .borrow(&guard)
                    .encoding
                    .execute(|e| *e.unwrap().clone());
                encoding
            };

            let default_pair = None;
            let pair = if let Some(arg) = cx.argument_opt(1) {
                if arg.downcast::<JsUndefined>().is_ok() {
                    default_pair
                } else {
                    arg.downcast_or_throw::<JsEncoding, _>(&mut cx).map(|e| {
                        let guard = cx.lock();
                        let encoding = e.borrow(&guard)
                            .encoding
                            .execute(|e| *e.unwrap().clone());
                        encoding
                    }).ok()
                }
            } else {
                default_pair
            };

            let default_add_special_tokens = true;
            let add_special_tokens = if let Some(arg) = cx.argument_opt(2) {
                if arg.downcast::<JsUndefined>().is_ok() {
                    default_add_special_tokens
                } else {
                    arg.downcast_or_throw::<JsBoolean, _>(&mut cx)?.value()
                }
            } else {
                default_add_special_tokens
            };

            let encoding = {
                let this = cx.this();
                let guard = cx.lock();
                let encoding = this.borrow(&guard)
                    .tokenizer.post_process(encoding, pair, add_special_tokens);
                encoding
            };
            let encoding = encoding
                .map_err(|e| cx.throw_error::<_, ()>(format!("{}", e)).unwrap_err())?;

            let mut js_encoding = JsEncoding::new::<_, JsEncoding, _>(&mut cx, vec![])?;
            let guard = cx.lock();
            js_encoding
                .borrow_mut(&guard)
                .encoding
                .to_owned(Box::new(encoding));

            Ok(js_encoding.upcast())
        }

        method getModel(mut cx) {
            // getModel(): Model

            let model = {
                let this = cx.this();
                let guard = cx.lock();
                let container = Container::from_ref(this.borrow(&guard).tokenizer.get_model());
                container
            };

            let mut js_model = JsModel::new::<_, JsModel, _>(&mut cx, vec![])?;
            let guard = cx.lock();
            js_model.borrow_mut(&guard).model = model;

            Ok(js_model.upcast())
        }

        method setModel(mut cx) {
            // setModel(model: JsModel)

            let running = {
                let this = cx.this();
                let guard = cx.lock();
                let count = std::sync::Arc::strong_count(&this.borrow(&guard).running_task);
                count
            };
            if running > 1 {
                println!("{} running tasks", running - 1);
                return cx.throw_error("Cannot modify the tokenizer while there are running tasks");
            }

            let mut model = cx.argument::<JsModel>(0)?;
            if let Some(instance) = {
                let guard = cx.lock();
                let mut model = model.borrow_mut(&guard);
                model.model.to_pointer()
            } {
                let mut this = cx.this();
                {
                    let guard = cx.lock();
                    let mut tokenizer = this.borrow_mut(&guard);
                    tokenizer.tokenizer.with_model(instance);
                }

                Ok(cx.undefined().upcast())
            } else {
                cx.throw_error("The Model is already being used in another Tokenizer")
            }
        }

        method getNormalizer(mut cx) {
            // getNormalizer(): Normalizer | undefined

            let normalizer = {
                let this = cx.this();
                let guard = cx.lock();
                let borrowed = this.borrow(&guard);
                let normalizer = borrowed.tokenizer.get_normalizer();
                normalizer.map(|normalizer| { Container::from_ref(normalizer) })
            };

            if let Some(normalizer) = normalizer {
                let mut js_normalizer = JsNormalizer::new::<_, JsNormalizer, _>(&mut cx, vec![])?;
                let guard = cx.lock();
                js_normalizer.borrow_mut(&guard).normalizer = normalizer;

                Ok(js_normalizer.upcast())
            } else {
                Ok(cx.undefined().upcast())
            }
        }

        method setNormalizer(mut cx) {
            // setNormalizer(normalizer: Normalizer)

            let running = {
                let this = cx.this();
                let guard = cx.lock();
                let count = std::sync::Arc::strong_count(&this.borrow(&guard).running_task);
                count
            };
            if running > 1 {
                println!("{} running tasks", running - 1);
                return cx.throw_error("Cannot modify the tokenizer while there are running tasks");
            }

            let mut normalizer = cx.argument::<JsNormalizer>(0)?;
            if let Some(instance) = {
                let guard = cx.lock();
                let mut normalizer = normalizer.borrow_mut(&guard);
                normalizer.normalizer.to_pointer()
            } {
                let mut this = cx.this();
                {
                    let guard = cx.lock();
                    let mut tokenizer = this.borrow_mut(&guard);
                    tokenizer.tokenizer.with_normalizer(instance);
                }

                Ok(cx.undefined().upcast())
            } else {
                cx.throw_error("The Normalizer is already being used in another Tokenizer")
            }
        }

        method getPreTokenizer(mut cx) {
            // getPreTokenizer(): PreTokenizer | undefined

            let pretok = {
                let this = cx.this();
                let guard = cx.lock();
                let borrowed = this.borrow(&guard);
                let pretok = borrowed.tokenizer.get_pre_tokenizer();
                pretok.map(|pretok| { Container::from_ref(pretok) })
            };

            if let Some(pretok) = pretok {
                let mut js_pretok = JsPreTokenizer::new::<_, JsPreTokenizer, _>(&mut cx, vec![])?;
                let guard = cx.lock();
                js_pretok.borrow_mut(&guard).pretok = pretok;

                Ok(js_pretok.upcast())
            } else {
                Ok(cx.undefined().upcast())
            }
        }

        method setPreTokenizer(mut cx) {
            // setPreTokenizer(pretokenizer: PreTokenizer)

            let running = {
                let this = cx.this();
                let guard = cx.lock();
                let count = std::sync::Arc::strong_count(&this.borrow(&guard).running_task);
                count
            };
            if running > 1 {
                println!("{} running tasks", running - 1);
                return cx.throw_error("Cannot modify the tokenizer while there are running tasks");
            }

            let mut pretok = cx.argument::<JsPreTokenizer>(0)?;
            if let Some(instance) = {
                let guard = cx.lock();
                let mut pretok = pretok.borrow_mut(&guard);
                pretok.pretok.to_pointer()
            } {
                let mut this = cx.this();
                {
                    let guard = cx.lock();
                    let mut tokenizer = this.borrow_mut(&guard);
                    tokenizer.tokenizer.with_pre_tokenizer(instance);
                }

                Ok(cx.undefined().upcast())
            } else {
                cx.throw_error("The PreTokenizer is already being used in another Tokenizer")
            }
        }

        method getPostProcessor(mut cx) {
            // getPostProcessor(): PostProcessor | undefined

            let processor = {
                let this = cx.this();
                let guard = cx.lock();
                let borrowed = this.borrow(&guard);
                let processor = borrowed.tokenizer.get_post_processor();
                processor.map(|processor| { Container::from_ref(processor) })
            };

            if let Some(processor) = processor {
                let mut js_processor = JsPostProcessor::new::<_, JsPostProcessor, _>(&mut cx, vec![])?;
                let guard = cx.lock();
                js_processor.borrow_mut(&guard).processor = processor;

                Ok(js_processor.upcast())
            } else {
                Ok(cx.undefined().upcast())
            }
        }

        method setPostProcessor(mut cx) {
            // setPostProcessor(processor: PostProcessor)

            let running = {
                let this = cx.this();
                let guard = cx.lock();
                let count = std::sync::Arc::strong_count(&this.borrow(&guard).running_task);
                count
            };
            if running > 1 {
                println!("{} running tasks", running - 1);
                return cx.throw_error("Cannot modify the tokenizer while there are running tasks");
            }

            let mut processor = cx.argument::<JsPostProcessor>(0)?;
            if let Some(instance) = {
                let guard = cx.lock();
                let mut processor = processor.borrow_mut(&guard);
                processor.processor.to_pointer()
            } {
                let mut this = cx.this();
                {
                    let guard = cx.lock();
                    let mut tokenizer = this.borrow_mut(&guard);
                    tokenizer.tokenizer.with_post_processor(instance);
                }

                Ok(cx.undefined().upcast())
            } else {
                cx.throw_error("The PostProcessor is already being used in another Tokenizer")
            }
        }

        method getDecoder(mut cx) {
            // getDecoder(): Decoder | undefined

            let decoder = {
                let this = cx.this();
                let guard = cx.lock();
                let borrowed = this.borrow(&guard);
                let decoder = borrowed.tokenizer.get_decoder();
                decoder.map(|decoder| { Container::from_ref(decoder) })
            };

            if let Some(decoder) = decoder {
                let mut js_decoder = JsDecoder::new::<_, JsDecoder, _>(&mut cx, vec![])?;
                let guard = cx.lock();
                js_decoder.borrow_mut(&guard).decoder = decoder;

                Ok(js_decoder.upcast())
            } else {
                Ok(cx.undefined().upcast())
            }
        }

        method setDecoder(mut cx) {
            // setDecoder(decoder: Decoder)

            let running = {
                let this = cx.this();
                let guard = cx.lock();
                let count = std::sync::Arc::strong_count(&this.borrow(&guard).running_task);
                count
            };
            if running > 1 {
                println!("{} running tasks", running - 1);
                return cx.throw_error("Cannot modify the tokenizer while there are running tasks");
            }

            let mut decoder = cx.argument::<JsDecoder>(0)?;
            if let Some(instance) = {
                let guard = cx.lock();
                let mut decoder = decoder.borrow_mut(&guard);
                decoder.decoder.to_pointer()
            } {
                let mut this = cx.this();
                {
                    let guard = cx.lock();
                    let mut tokenizer = this.borrow_mut(&guard);
                    tokenizer.tokenizer.with_decoder(instance);
                }

                Ok(cx.undefined().upcast())
            } else {
                cx.throw_error("The Decoder is already being used in another Tokenizer")
            }
        }
    }
}

pub fn register(m: &mut ModuleContext, prefix: &str) -> Result<(), neon::result::Throw> {
    m.export_class::<JsAddedToken>(&format!("{}_AddedToken", prefix))?;
    m.export_class::<JsTokenizer>(&format!("{}_Tokenizer", prefix))?;
    Ok(())
}
