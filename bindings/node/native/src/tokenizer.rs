extern crate tokenizers as tk;

use crate::models::*;
use crate::tasks::tokenizer::{EncodeTask, WorkingTokenizer};
use neon::prelude::*;

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
                count
            };
            Ok(cx.number(running as f64).upcast())
        }

        method getVocabSize(mut cx) {
            // getVocabSize(withAddedTokens: bool = true)
            let mut with_added_tokens = true;
            if let Some(args) = cx.argument_opt(0) {
                with_added_tokens = args.downcast::<JsBoolean>().or_throw(&mut cx)?.value() as bool;
            }

            let mut this = cx.this();
            let guard = cx.lock();
            let size = this.borrow_mut(&guard).tokenizer.get_vocab_size(with_added_tokens);

            Ok(cx.number(size as f64).upcast())
        }

        method withModel(mut cx) {
            let running = {
                let this = cx.this();
                let guard = cx.lock();
                let count = std::sync::Arc::strong_count(&this.borrow(&guard).running_task);
                count
            };
            if running > 1 {
                println!("{} running tasks", running);
                return cx.throw_error("Cannot modify the tokenizer while there are running tasks");
            }

            // with_model(model: JsModel)
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

        method encode(mut cx) {
            // encode(sentence: String, pair: String | null = null, __callback: (err, encoding) -> void)
            let sentence = cx.argument::<JsString>(0)?.value();
            let mut pair: Option<String> = None;
            if let Some(args) = cx.argument_opt(1) {
                if let Ok(p) = args.downcast::<JsString>() {
                    pair = Some(p.value());
                } else if let Err(_) = args.downcast::<JsNull>() {
                    return cx.throw_error("Second arg must be of type `String | null`");
                }
            }
            let callback = cx.argument::<JsFunction>(2)?;

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

            let task = EncodeTask::Single(worker, Some(input));
            task.schedule(callback);
            Ok(cx.undefined().upcast())
        }

        method encodeBatch(mut cx) {
            // type EncodeInput = (String | [String, String])[]
            // encode_batch(sentences: EncodeInput[], __callback: (err, encodings) -> void)
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
            let callback = cx.argument::<JsFunction>(1)?;

            let worker = {
                let this = cx.this();
                let guard = cx.lock();
                let worker = this.borrow(&guard).prepare_for_task();
                worker
            };

            let task = EncodeTask::Batch(worker, Some(inputs));
            task.schedule(callback);
            Ok(cx.undefined().upcast())
        }
    }
}

pub fn register(m: &mut ModuleContext, prefix: &str) -> Result<(), neon::result::Throw> {
    m.export_class::<JsTokenizer>(&format!("{}_Tokenizer", prefix))?;
    Ok(())
}
