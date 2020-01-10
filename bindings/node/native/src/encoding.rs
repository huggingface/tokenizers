extern crate tokenizers as tk;

use tk::tokenizer::PaddingDirection;

use crate::utils::Container;
use neon::prelude::*;

/// Encoding
pub struct Encoding {
    pub encoding: Container<tk::tokenizer::Encoding>,
}

declare_types! {
    pub class JsEncoding for Encoding {
        init(_) {
            // This should never be called from JavaScript
            Ok(Encoding {
                encoding: Container::Empty
            })
        }

        method getIds(mut cx) {
            // getIds(): number[]

            let this = cx.this();
            let guard = cx.lock();
            let ids = this.borrow(&guard).encoding.execute(|encoding| {
                encoding.unwrap().get_ids().to_vec()
            });
            let js_ids = JsArray::new(&mut cx, ids.len() as u32);
            for (i, id) in ids.into_iter().enumerate() {
                let n = JsNumber::new(&mut cx, id as f64);
                js_ids.set(&mut cx, i as u32, n)?;
            }

            Ok(js_ids.upcast())
        }

        method getTypeIds(mut cx) {
            // getTypeIds(): number[]

            let this = cx.this();
            let guard = cx.lock();
            let ids = this.borrow(&guard).encoding.execute(|encoding| {
                encoding.unwrap().get_type_ids().to_vec()
            });
            let js_ids = JsArray::new(&mut cx, ids.len() as u32);
            for (i, id) in ids.into_iter().enumerate() {
                let n = JsNumber::new(&mut cx, id as f64);
                js_ids.set(&mut cx, i as u32, n)?;
            }

            Ok(js_ids.upcast())
        }

        method getAttentionMask(mut cx) {
            // getAttentionMask(): number[]

            let this = cx.this();
            let guard = cx.lock();
            let ids = this.borrow(&guard).encoding.execute(|encoding| {
                encoding.unwrap().get_attention_mask().to_vec()
            });
            let js_ids = JsArray::new(&mut cx, ids.len() as u32);
            for (i, id) in ids.into_iter().enumerate() {
                let n = JsNumber::new(&mut cx, id as f64);
                js_ids.set(&mut cx, i as u32, n)?;
            }

            Ok(js_ids.upcast())
        }

        method getSpecialTokensMask(mut cx) {
            // getSpecialTokensMask(): number[]

            let this = cx.this();
            let guard = cx.lock();
            let ids = this.borrow(&guard).encoding.execute(|encoding| {
                encoding.unwrap().get_special_tokens_mask().to_vec()
            });
            let js_ids = JsArray::new(&mut cx, ids.len() as u32);
            for (i, id) in ids.into_iter().enumerate() {
                let n = JsNumber::new(&mut cx, id as f64);
                js_ids.set(&mut cx, i as u32, n)?;
            }

            Ok(js_ids.upcast())
        }

        method getTokens(mut cx) {
            // getTokens(): string[]

            let this = cx.this();
            let guard = cx.lock();
            let tokens = this.borrow(&guard).encoding.execute(|encoding| {
                encoding.unwrap().get_tokens().to_vec()
            });
            let js_tokens = JsArray::new(&mut cx, tokens.len() as u32);
            for (i, token) in tokens.into_iter().enumerate() {
                let n = JsString::new(&mut cx, token);
                js_tokens.set(&mut cx, i as u32, n)?;
            }

            Ok(js_tokens.upcast())
        }

        method getOffsets(mut cx) {
            // getOffsets(): [number, number][]

            let this = cx.this();
            let guard = cx.lock();
            let offsets = this.borrow(&guard).encoding.execute(|encoding| {
                encoding.unwrap().get_offsets().to_vec()
            });
            let js_offsets = JsArray::new(&mut cx, offsets.len() as u32);
            for (i, offsets) in offsets.into_iter().enumerate() {
                let n = JsArray::new(&mut cx, 2);
                let o_0 = JsNumber::new(&mut cx, offsets.0 as f64);
                let o_1 = JsNumber::new(&mut cx, offsets.1 as f64);
                n.set(&mut cx, 0, o_0)?;
                n.set(&mut cx, 1, o_1)?;
                js_offsets.set(&mut cx, i as u32, n)?;
            }

            Ok(js_offsets.upcast())
        }

        method getOverflowing(mut cx) {
            // getOverflowing(): Encoding | undefined;

            let this = cx.this();
            let guard = cx.lock();

            let overflowing = this.borrow(&guard).encoding.execute(|encoding| {
                encoding.unwrap().get_overflowing().cloned()
            });

            if let Some(overflowing) = overflowing {
                let mut js_overflowing = JsEncoding::new::<_, JsEncoding, _>(&mut cx, vec![])?;

                // Set the content
                let guard = cx.lock();
                js_overflowing.borrow_mut(&guard).encoding.to_owned(Box::new(overflowing));

                Ok(js_overflowing.upcast())
            } else {
                Ok(cx.undefined().upcast())
            }
        }

        method pad(mut cx) {
            // pad(length: number, options?: {
            //   direction?: 'left' | 'right' = 'right',
            //   padId?: number = 0,
            //   padTypeId?: number = 0,
            //   padToken?: string = "[PAD]"
            // }

            let length = cx.argument::<JsNumber>(0)?.value() as usize;
            let mut direction = PaddingDirection::Right;
            let mut pad_id = 0;
            let mut pad_type_id = 0;
            let mut pad_token = String::from("[PAD]");

            let options = cx.argument_opt(1);
            if let Some(options) = options {
                if let Ok(options) = options.downcast::<JsObject>() {
                    if let Ok(dir) = options.get(&mut cx, "direction") {
                        if let Err(_) = dir.downcast::<JsUndefined>() {
                            let dir = dir.downcast::<JsString>().or_throw(&mut cx)?.value();
                            match &dir[..] {
                                "right" => direction = PaddingDirection::Right,
                                "left" => direction = PaddingDirection::Left,
                                _ => return cx.throw_error("direction can be 'right' or 'left'"),
                            }
                        }
                    }
                    if let Ok(pid) = options.get(&mut cx, "padId") {
                        if let Err(_) = pid.downcast::<JsUndefined>() {
                            pad_id = pid.downcast::<JsNumber>().or_throw(&mut cx)?.value() as u32;
                        }
                    }
                    if let Ok(pid) = options.get(&mut cx, "padTypeId") {
                        if let Err(_) = pid.downcast::<JsUndefined>() {
                            pad_type_id = pid.downcast::<JsNumber>().or_throw(&mut cx)?.value() as u32;
                        }
                    }
                    if let Ok(token) = options.get(&mut cx, "padToken") {
                        if let Err(_) = token.downcast::<JsUndefined>() {
                            pad_token = token.downcast::<JsString>().or_throw(&mut cx)?.value();
                        }
                    }
                }
            }

            let mut this = cx.this();
            let guard = cx.lock();
            this.borrow_mut(&guard).encoding.execute_mut(|encoding| {
                encoding.unwrap().pad(length, pad_id, pad_type_id, &pad_token, &direction);
            });

            Ok(cx.undefined().upcast())
        }

        method truncate(mut cx) {
            // truncate(length: number, stride: number = 0)

            let length = cx.argument::<JsNumber>(0)?.value() as usize;
            let mut stride = 0;
            if let Some(args) = cx.argument_opt(1) {
                stride = args.downcast::<JsNumber>().or_throw(&mut cx)?.value() as usize;
            }

            let mut this = cx.this();
            let guard = cx.lock();
            this.borrow_mut(&guard).encoding.execute_mut(|encoding| {
                encoding.unwrap().truncate(length, stride);
            });

            Ok(cx.undefined().upcast())
        }
    }
}
