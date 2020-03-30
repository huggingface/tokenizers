extern crate tokenizers as tk;

use crate::container::Container;
use crate::decoders::JsDecoder;
use crate::encoding::JsEncoding;
use crate::models::JsModel;
use crate::normalizers::JsNormalizer;
use crate::pre_tokenizers::JsPreTokenizer;
use crate::processors::JsPostProcessor;
use crate::tasks::tokenizer::{DecodeTask, EncodeTask, EncodeTokenizedTask, WorkingTokenizer};
use crate::trainers::JsTrainer;
use neon::prelude::*;

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
            // encode(
            //   sentence: String,
            //   pair: String | null,
            //   add_special_tokens: boolean,
            //   __callback: (err, encoding) -> void
            // )
            let sentence = cx.argument::<JsString>(0)?.value();
            let mut pair: Option<String> = None;
            if let Some(args) = cx.argument_opt(1) {
                if let Ok(p) = args.downcast::<JsString>() {
                    pair = Some(p.value());
                } else if args.downcast::<JsNull>().is_err() {
                    return cx.throw_error("Second arg must be of type `String | null`");
                }
            }
            let add_special_tokens = cx.argument::<JsBoolean>(2)?.value();
            let callback = cx.argument::<JsFunction>(3)?;

            let input = if let Some(pair) = pair {
                tk::tokenizer::EncodeInput::Dual(sentence, pair)
            } else {
                tk::tokenizer::EncodeInput::Single(sentence)
            };

            let worker = {
                let this = cx.this();
                let guard = cx.lock();
                let worker = this.borrow(&guard).prepare_for_task();
                worker
            };

            let task = EncodeTask::Single(worker, Some(input), add_special_tokens);
            task.schedule(callback);
            Ok(cx.undefined().upcast())
        }

        method encodeBatch(mut cx) {
            // type EncodeInput = (String | [String, String])[]
            // encode_batch(
            //   sentences: EncodeInput[],
            //   add_special_tokens: boolean,
            //   __callback: (err, encodings) -> void
            // )
            let inputs = cx.argument::<JsArray>(0)?.to_vec(&mut cx)?;
            let inputs = inputs.into_iter().map(|value| {
                if let Ok(s) = value.downcast::<JsString>() {
                    Ok(tk::tokenizer::EncodeInput::Single(s.value()))
                } else if let Ok(arr) = value.downcast::<JsArray>() {
                    if arr.len() != 2 {
                        cx.throw_error("Input must be an array of `String | [String, String]`")
                    } else {
                        Ok(tk::tokenizer::EncodeInput::Dual(
                            arr.get(&mut cx, 0)?
                                .downcast::<JsString>()
                                .or_throw(&mut cx)?
                                .value(),
                            arr.get(&mut cx, 1)?
                                .downcast::<JsString>()
                                .or_throw(&mut cx)?
                                .value())
                        )
                    }
                } else {
                    cx.throw_error("Input must be an array of `String | [String, String]`")
                }
            }).collect::<NeonResult<Vec<_>>>()?;
            let add_special_tokens = cx.argument::<JsBoolean>(1)?.value();
            let callback = cx.argument::<JsFunction>(2)?;

            let worker = {
                let this = cx.this();
                let guard = cx.lock();
                let worker = this.borrow(&guard).prepare_for_task();
                worker
            };

            let task = EncodeTask::Batch(worker, Some(inputs), add_special_tokens);
            task.schedule(callback);
            Ok(cx.undefined().upcast())
        }

        method encodeTokenized(mut cx) {
            /// encodeTokenized(
            ///   sequence: (String | [String, [number, number]])[],
            ///   typeId?: number = 0,
            ///   callback: (err, Encoding)
            /// )

            let sequence = cx.argument::<JsArray>(0)?.to_vec(&mut cx)?;

            let type_arg = cx.argument::<JsValue>(1)?;
            let type_id = if type_arg.downcast::<JsUndefined>().is_err() {
                type_arg.downcast_or_throw::<JsNumber, _>(&mut cx)?.value() as u32
            } else {
                0
            };

            enum Mode {
                NoOffsets,
                Offsets,
            };
            let mode  = sequence.iter().next().map(|item| {
                if item.downcast::<JsString>().is_ok() {
                    Ok(Mode::NoOffsets)
                } else if item.downcast::<JsArray>().is_ok() {
                    Ok(Mode::Offsets)
                } else {
                    Err("Input must be (String | [String, [number, number]])[]")
                }
            })
            .unwrap()
            .map_err(|e| cx.throw_error::<_, ()>(e.to_string()).unwrap_err())?;

            let mut total_len = 0;
            let sequence = sequence.iter().map(|item| match mode {
                Mode::NoOffsets => {
                    let s = item.downcast::<JsString>().or_throw(&mut cx)?.value();
                    let len = s.chars().count();
                    total_len += len;
                    Ok((s, (total_len - len, total_len)))
                },
                Mode::Offsets => {
                    let tuple = item.downcast::<JsArray>().or_throw(&mut cx)?;
                    let s = tuple.get(&mut cx, 0)?
                        .downcast::<JsString>()
                        .or_throw(&mut cx)?
                        .value();
                    let offsets = tuple.get(&mut cx, 1)?
                        .downcast::<JsArray>()
                        .or_throw(&mut cx)?;
                    let (start, end) = (
                        offsets.get(&mut cx, 0)?
                            .downcast::<JsNumber>()
                            .or_throw(&mut cx)?
                            .value() as usize,
                        offsets.get(&mut cx, 1)?
                            .downcast::<JsNumber>().
                            or_throw(&mut cx)?
                            .value() as usize,
                    );
                    Ok((s, (start, end)))
                }
            }).collect::<Result<Vec<_>, _>>()?;
            let callback = cx.argument::<JsFunction>(2)?;

            let worker = {
                let this = cx.this();
                let guard = cx.lock();
                let worker = this.borrow(&guard).prepare_for_task();
                worker
            };

            let task = EncodeTokenizedTask::Single(worker, Some(sequence), type_id);
            task.schedule(callback);
            Ok(cx.undefined().upcast())
        }

        method encodeTokenizedBatch(mut cx) {
            /// encodeTokenizedBatch(
            ///   sequences: (String | [String, [number, number]])[][],
            ///   typeId?: number = 0,
            ///   callback: (err, Encoding)
            /// )

            let sequences = cx.argument::<JsArray>(0)?.to_vec(&mut cx)?;

            let type_arg = cx.argument::<JsValue>(1)?;
            let type_id = if type_arg.downcast::<JsUndefined>().is_err() {
                type_arg.downcast_or_throw::<JsNumber, _>(&mut cx)?.value() as u32
            } else {
                0
            };

            enum Mode {
                NoOffsets,
                Offsets,
            };
            let mode  = sequences.iter().next().map(|sequence| {
                if let Ok(sequence) = sequence.downcast::<JsArray>().or_throw(&mut cx) {
                    sequence.to_vec(&mut cx).ok().map(|s| s.iter().next().map(|item| {
                        if item.downcast::<JsString>().is_ok() {
                            Some(Mode::NoOffsets)
                        } else if item.downcast::<JsArray>().is_ok() {
                            Some(Mode::Offsets)
                        } else {
                            None
                        }
                    }).flatten()).flatten()
                } else {
                    None
                }
            })
            .flatten()
            .ok_or_else(||
                cx.throw_error::<_, ()>(
                    "Input must be (String | [String, [number, number]])[]"
                ).unwrap_err()
            )?;

            let sequences = sequences.into_iter().map(|sequence| {
                let mut total_len = 0;
                sequence.downcast::<JsArray>().or_throw(&mut cx)?
                    .to_vec(&mut cx)?
                    .into_iter()
                    .map(|item| match mode {
                        Mode::NoOffsets => {
                            let s = item.downcast::<JsString>().or_throw(&mut cx)?.value();
                            let len = s.chars().count();
                            total_len += len;
                            Ok((s, (total_len - len, total_len)))
                        },
                        Mode::Offsets => {
                            let tuple = item.downcast::<JsArray>().or_throw(&mut cx)?;
                            let s = tuple.get(&mut cx, 0)?
                                .downcast::<JsString>()
                                .or_throw(&mut cx)?
                                .value();
                            let offsets = tuple.get(&mut cx, 1)?
                                .downcast::<JsArray>()
                                .or_throw(&mut cx)?;
                            let (start, end) = (
                                offsets.get(&mut cx, 0)?
                                    .downcast::<JsNumber>()
                                    .or_throw(&mut cx)?
                                    .value() as usize,
                                offsets.get(&mut cx, 1)?
                                    .downcast::<JsNumber>().
                                    or_throw(&mut cx)?
                                    .value() as usize,
                            );
                            Ok((s, (start, end)))
                        }
                    }).collect::<Result<Vec<_>, _>>()
                })
                .collect::<Result<Vec<_>, _>>()?;
            let callback = cx.argument::<JsFunction>(2)?;

            let worker = {
                let this = cx.this();
                let guard = cx.lock();
                let worker = this.borrow(&guard).prepare_for_task();
                worker
            };

            let task = EncodeTokenizedTask::Batch(worker, Some(sequences), type_id);
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
