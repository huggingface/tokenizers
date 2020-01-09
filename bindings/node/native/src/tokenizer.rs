extern crate tokenizers as tk;

use crate::encoding::*;
use crate::models::*;
use neon::prelude::*;

/// Tokenizer
pub struct Tokenizer {
    tokenizer: tk::tokenizer::Tokenizer,
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
                    tokenizer: tk::tokenizer::Tokenizer::new(instance)
                })
            } else {
                cx.throw_error("The Model is already being used in another Tokenizer")
            }
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
            // encode(sentence: String, pair?: String): Encoding
            let sentence = cx.argument::<JsString>(0)?.value();
            let mut pair: Option<String> = None;
            if let Some(args) = cx.argument_opt(1) {
                pair = Some(args.downcast::<JsString>().or_throw(&mut cx)?.value());
            }

            let input = if let Some(pair) = pair {
                tk::tokenizer::EncodeInput::Dual(sentence, pair)
            } else {
                tk::tokenizer::EncodeInput::Single(sentence)
            };

            let encoding = {
                let this = cx.this();
                let guard = cx.lock();
                let res = this.borrow(&guard).tokenizer.encode(input);
                res.map_err(|e| cx.throw_error::<_, ()>(format!("{}", e)).unwrap_err())?
            };

            let mut js_encoding = JsEncoding::new::<_, JsEncoding, _>(&mut cx, vec![])?;
            // Set the actual encoding
            let guard = cx.lock();
            js_encoding.borrow_mut(&guard).encoding.to_owned(Box::new(encoding));

            Ok(js_encoding.upcast())
        }

        method encodeBatch(mut cx) {
            // type EncodeInput = (String | [String, String])[]
            // encode_batch(sentences: EncodeInput[]): Encoding[]
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

            let encodings = {
                let this = cx.this();
                let guard = cx.lock();
                let res = this.borrow(&guard).tokenizer.encode_batch(inputs);
                res.map_err(|e| cx.throw_error::<_, ()>(format!("{}", e)).unwrap_err())?
            };

            let result = JsArray::new(&mut cx, encodings.len() as u32);
            for (i, encoding) in encodings.into_iter().enumerate() {
                let mut js_encoding = JsEncoding::new::<_, JsEncoding, _>(&mut cx, vec![])?;

                // Set the actual encoding
                let guard = cx.lock();
                js_encoding.borrow_mut(&guard).encoding.to_owned(Box::new(encoding));

                result.set(&mut cx, i as u32, js_encoding)?;
            }

            Ok(result.upcast())
        }
    }
}

pub fn register(m: &mut ModuleContext, prefix: &str) -> Result<(), neon::result::Throw> {
    m.export_class::<JsTokenizer>(&format!("{}_Tokenizer", prefix))?;
    Ok(())
}
