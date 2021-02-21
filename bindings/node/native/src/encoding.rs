extern crate tokenizers as tk;

use crate::extraction::*;
use crate::tokenizer::PaddingParams;
use neon::prelude::*;

/// Encoding
pub struct Encoding {
    pub encoding: Option<tk::tokenizer::Encoding>,
}

declare_types! {
    pub class JsEncoding for Encoding {
        init(_) {
            // This should never be called from JavaScript
            Ok(Encoding { encoding: None })
        }

        method getLength(mut cx) {
            let this = cx.this();
            let guard = cx.lock();
            let length = this.borrow(&guard)
                .encoding.as_ref().expect("Uninitialized Encoding")
                .get_ids()
                .len();

            Ok(cx.number(length as f64).upcast())
        }

        method getNSequences(mut cx) {
            let this = cx.this();
            let guard = cx.lock();
            let n = this.borrow(&guard)
                .encoding.as_ref().expect("Uninitialized Encoding")
                .n_sequences();

            Ok(cx.number(n as f64).upcast())
        }

        method setSequenceId(mut cx) {
            let seq_id = cx.extract::<usize>(0)?;

            let mut this = cx.this();
            let guard = cx.lock();
            this.borrow_mut(&guard)
                .encoding.as_mut().expect("Uninitialized Encoding")
                .set_sequence_id(seq_id);

            Ok(cx.undefined().upcast())
        }

        method getIds(mut cx) {
            // getIds(): number[]

            let this = cx.this();
            let guard = cx.lock();
            let ids = this.borrow(&guard)
                .encoding.as_ref().expect("Uninitialized Encoding")
                .get_ids()
                .to_vec();

            Ok(neon_serde::to_value(&mut cx, &ids)?)
        }

        method getTypeIds(mut cx) {
            // getTypeIds(): number[]

            let this = cx.this();
            let guard = cx.lock();
            let ids = this.borrow(&guard)
                .encoding.as_ref().expect("Uninitialized Encoding")
                .get_type_ids()
                .to_vec();

            Ok(neon_serde::to_value(&mut cx, &ids)?)
        }

        method getAttentionMask(mut cx) {
            // getAttentionMask(): number[]

            let this = cx.this();
            let guard = cx.lock();
            let ids = this.borrow(&guard)
                .encoding.as_ref().expect("Uninitialized Encoding")
                .get_attention_mask()
                .to_vec();

            Ok(neon_serde::to_value(&mut cx, &ids)?)
        }

        method getSpecialTokensMask(mut cx) {
            // getSpecialTokensMask(): number[]

            let this = cx.this();
            let guard = cx.lock();
            let ids = this.borrow(&guard)
                .encoding.as_ref().expect("Uninitialized Encoding")
                .get_special_tokens_mask()
                .to_vec();

            Ok(neon_serde::to_value(&mut cx, &ids)?)
        }

        method getTokens(mut cx) {
            // getTokens(): string[]

            let this = cx.this();
            let guard = cx.lock();
            let tokens = this.borrow(&guard)
                .encoding.as_ref().expect("Uninitialized Encoding")
                .get_tokens()
                .to_vec();

            Ok(neon_serde::to_value(&mut cx, &tokens)?)
        }

        method getWordIds(mut cx) {
            // getWordIds(): (number | undefined)[]

            let this = cx.this();
            let guard = cx.lock();
            let ids = this.borrow(&guard)
                .encoding.as_ref().expect("Uninitialized Encoding")
                .get_word_ids()
                .to_vec();

            Ok(neon_serde::to_value(&mut cx, &ids)?)
        }

        method getSequenceIds(mut cx) {
            // getSequenceIds(): (number | undefined)[]

            let this = cx.this();
            let guard = cx.lock();
            let ids = this.borrow(&guard)
                .encoding.as_ref().expect("Uninitialized Encoding")
                .get_sequence_ids();

            Ok(neon_serde::to_value(&mut cx, &ids)?)
        }

        method getOffsets(mut cx) {
            // getOffsets(): [number, number][]

            let this = cx.this();
            let guard = cx.lock();
            let offsets = this.borrow(&guard)
                .encoding.as_ref().expect("Uninitialized Encoding")
                .get_offsets()
                .to_vec();
            let js_offsets = neon_serde::to_value(&mut cx, &offsets)?;

            Ok(js_offsets)
        }

        method getOverflowing(mut cx) {
            // getOverflowing(): Encoding[]

            let this = cx.this();
            let guard = cx.lock();

            let overflowings = this.borrow(&guard)
                .encoding.as_ref().expect("Uninitialized Encoding")
                .get_overflowing()
                .clone();
            let js_overflowings = JsArray::new(&mut cx, overflowings.len() as u32);

            for (index, overflowing) in overflowings.iter().enumerate() {
                let mut js_overflowing = JsEncoding::new::<_, JsEncoding, _>(&mut cx, vec![])?;

                // Set the content
                let guard = cx.lock();
                js_overflowing.borrow_mut(&guard).encoding = Some(overflowing.clone());

                js_overflowings.set(&mut cx, index as u32, js_overflowing)?;
            }

            Ok(js_overflowings.upcast())
        }

        method wordToTokens(mut cx) {
            // wordToTokens(word: number, seqId: number = 0): [number, number] | undefined

            let word = cx.extract::<u32>(0)?;
            let seq_id = cx.extract_opt::<usize>(1)?.unwrap_or(0);

            let this = cx.this();
            let guard = cx.lock();

            let res = this.borrow(&guard)
                .encoding.as_ref().expect("Uninitialized Encoding")
                .word_to_tokens(word, seq_id);

            if let Some(tokens) = res {
                Ok(neon_serde::to_value(&mut cx, &tokens)?)
            } else {
                Ok(cx.undefined().upcast())
            }
        }

        method wordToChars(mut cx) {
            // wordToChars(word: number, seqId: number = 0): [number, number] | undefined

            let word = cx.extract::<u32>(0)?;
            let seq_id = cx.extract_opt::<usize>(1)?.unwrap_or(0);

            let this = cx.this();
            let guard = cx.lock();

            let res = this.borrow(&guard)
                .encoding.as_ref().expect("Uninitialized Encoding")
                .word_to_chars(word, seq_id);

            if let Some(offsets) = res {
                Ok(neon_serde::to_value(&mut cx, &offsets)?)
            } else {
                Ok(cx.undefined().upcast())
            }
        }

        method tokenToSequence(mut cx) {
            // tokenToSequence(token: number): number | undefined

            let token = cx.extract::<usize>(0)?;

            let this = cx.this();
            let guard = cx.lock();

            let res = this.borrow(&guard)
                .encoding.as_ref().expect("Uninitialized Encoding")
                .token_to_sequence(token);

            if let Some(seq) = res {
                Ok(neon_serde::to_value(&mut cx, &seq)?)
            } else {
                Ok(cx.undefined().upcast())
            }
        }

        method tokenToChars(mut cx) {
            // tokenToChars(token: number): [number, number] [number, [number, number]] | undefined

            let token = cx.extract::<usize>(0)?;

            let this = cx.this();
            let guard = cx.lock();

            let res = this.borrow(&guard)
                .encoding.as_ref().expect("Uninitialized Encoding")
                .token_to_chars(token);

            if let Some((_, offsets)) = res {
                Ok(neon_serde::to_value(&mut cx, &offsets)?)
            } else {
                Ok(cx.undefined().upcast())
            }
        }

        method tokenToWord(mut cx) {
            // tokenToWord(token: number): number | [number, number] | undefined

            let token = cx.argument::<JsNumber>(0)?.value() as usize;

            let this = cx.this();
            let guard = cx.lock();

            let res = this.borrow(&guard)
                .encoding.as_ref().expect("Uninitialized Encoding")
                .token_to_word(token);

            if let Some((_, index)) = res {
                Ok(cx.number(index as f64).upcast())
            } else {
                Ok(cx.undefined().upcast())
            }
        }

        method charToToken(mut cx) {
            // charToToken(pos: number, seqId: number = 0): number | undefined

            let pos = cx.extract::<usize>(0)?;
            let seq_id = cx.extract_opt::<usize>(1)?.unwrap_or(0);

            let this = cx.this();
            let guard = cx.lock();
            let index = this.borrow(&guard)
                .encoding.as_ref().expect("Uninitialized Encoding")
                .char_to_token(pos, seq_id);

            if let Some(index) = index {
                Ok(cx.number(index as f64).upcast())
            } else {
                Ok(cx.undefined().upcast())
            }
        }

        method charToWord(mut cx) {
            // charToWord(pos: number, seqId: number = 0): number | undefined

            let pos = cx.extract::<usize>(0)?;
            let seq_id = cx.extract_opt::<usize>(1)?.unwrap_or(0);

            let this = cx.this();
            let guard = cx.lock();
            let index = this.borrow(&guard)
                .encoding.as_ref().expect("Uninitialized Encoding")
                .char_to_word(pos, seq_id);

            if let Some(index) = index {
                Ok(cx.number(index as f64).upcast())
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
            let length = cx.extract::<usize>(0)?;
            let params = cx.extract_opt::<PaddingParams>(1)?
                .map_or_else(tk::PaddingParams::default, |p| p.0);

            let mut this = cx.this();
            let guard = cx.lock();
            this.borrow_mut(&guard)
                .encoding.as_mut().expect("Uninitialized Encoding")
                .pad(
                    length,
                    params.pad_id,
                    params.pad_type_id,
                    &params.pad_token,
                    params.direction
                );

            Ok(cx.undefined().upcast())
        }

        method truncate(mut cx) {
            // truncate(length: number, stride: number = 0)

            let length = cx.extract::<usize>(0)?;
            let stride = cx.extract_opt::<usize>(1)?.unwrap_or(0);

            let mut this = cx.this();
            let guard = cx.lock();
            this.borrow_mut(&guard)
                .encoding.as_mut().expect("Uninitialized Encoding")
                .truncate(length, stride);

            Ok(cx.undefined().upcast())
        }
    }
}
