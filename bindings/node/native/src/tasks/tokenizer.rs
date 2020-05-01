extern crate tokenizers as tk;

use crate::encoding::*;
use neon::prelude::*;
use tk::tokenizer::{EncodeInput, Encoding, Tokenizer};

pub struct WorkingTokenizer {
    _arc: std::sync::Arc<()>,
    ptr: *const Tokenizer,
}
impl WorkingTokenizer {
    /// This is unsafe because the caller must ensure that the given tokenizer
    /// wont be modified for the duration of the task. We keep an arc here to let the
    /// caller know when we are done with our pointer on Tokenizer
    pub unsafe fn new(tokenizer: &Tokenizer, arc: std::sync::Arc<()>) -> Self {
        WorkingTokenizer {
            _arc: arc,
            ptr: tokenizer as *const _,
        }
    }
}
unsafe impl Send for WorkingTokenizer {}

pub enum EncodeTask {
    Single(WorkingTokenizer, Option<EncodeInput>, bool),
    Batch(WorkingTokenizer, Option<Vec<EncodeInput>>, bool),
}

pub enum EncodeOutput {
    Single(Encoding),
    Batch(Vec<Encoding>),
}

impl Task for EncodeTask {
    type Output = EncodeOutput;
    type Error = String;
    type JsEvent = JsValue;

    fn perform(&self) -> Result<Self::Output, Self::Error> {
        match self {
            EncodeTask::Single(worker, input, add_special_tokens) => {
                let mut input: Option<EncodeInput> =
                    unsafe { std::ptr::replace(input as *const _ as *mut _, None) };
                let tokenizer: &Tokenizer = unsafe { &*worker.ptr };
                tokenizer
                    .encode(
                        input.take().ok_or("No provided input")?,
                        *add_special_tokens,
                    )
                    .map_err(|e| format!("{}", e))
                    .map(|encoding| EncodeOutput::Single(encoding))
            }
            EncodeTask::Batch(worker, input, add_special_tokens) => {
                let mut input: Option<Vec<EncodeInput>> =
                    unsafe { std::ptr::replace(input as *const _ as *mut _, None) };
                let tokenizer: &Tokenizer = unsafe { &*worker.ptr };
                tokenizer
                    .encode_batch(
                        input.take().ok_or("No provided input")?,
                        *add_special_tokens,
                    )
                    .map_err(|e| format!("{}", e))
                    .map(|encodings| EncodeOutput::Batch(encodings))
            }
        }
    }

    fn complete(
        self,
        mut cx: TaskContext,
        result: Result<Self::Output, Self::Error>,
    ) -> JsResult<Self::JsEvent> {
        match result.map_err(|e| cx.throw_error::<_, ()>(e).unwrap_err())? {
            EncodeOutput::Single(encoding) => {
                let mut js_encoding = JsEncoding::new::<_, JsEncoding, _>(&mut cx, vec![])?;
                // Set the actual encoding
                let guard = cx.lock();
                js_encoding
                    .borrow_mut(&guard)
                    .encoding
                    .to_owned(Box::new(encoding));

                Ok(js_encoding.upcast())
            }
            EncodeOutput::Batch(encodings) => {
                let result = JsArray::new(&mut cx, encodings.len() as u32);
                for (i, encoding) in encodings.into_iter().enumerate() {
                    let mut js_encoding = JsEncoding::new::<_, JsEncoding, _>(&mut cx, vec![])?;

                    // Set the actual encoding
                    let guard = cx.lock();
                    js_encoding
                        .borrow_mut(&guard)
                        .encoding
                        .to_owned(Box::new(encoding));

                    result.set(&mut cx, i as u32, js_encoding)?;
                }
                Ok(result.upcast())
            }
        }
    }
}

pub enum DecodeTask {
    Single(WorkingTokenizer, Vec<u32>, bool),
    Batch(WorkingTokenizer, Vec<Vec<u32>>, bool),
}

pub enum DecodeOutput {
    Single(String),
    Batch(Vec<String>),
}

impl Task for DecodeTask {
    type Output = DecodeOutput;
    type Error = String;
    type JsEvent = JsValue;

    fn perform(&self) -> Result<Self::Output, Self::Error> {
        match self {
            DecodeTask::Single(worker, ids, skip_special_tokens) => {
                let tokenizer: &Tokenizer = unsafe { &*worker.ptr };
                tokenizer
                    .decode(ids.to_vec(), *skip_special_tokens)
                    .map_err(|e| format!("{}", e))
                    .map(DecodeOutput::Single)
            }
            DecodeTask::Batch(worker, ids, skip_special_tokens) => {
                let tokenizer: &Tokenizer = unsafe { &*worker.ptr };
                tokenizer
                    .decode_batch(ids.to_vec(), *skip_special_tokens)
                    .map_err(|e| format!("{}", e))
                    .map(DecodeOutput::Batch)
            }
        }
    }

    fn complete(
        self,
        mut cx: TaskContext,
        result: Result<Self::Output, Self::Error>,
    ) -> JsResult<Self::JsEvent> {
        match result.map_err(|e| cx.throw_error::<_, ()>(e).unwrap_err())? {
            DecodeOutput::Single(string) => Ok(cx.string(string).upcast()),
            DecodeOutput::Batch(strings) => {
                let result = JsArray::new(&mut cx, strings.len() as u32);
                for (i, string) in strings.into_iter().enumerate() {
                    let js_string = cx.string(string);
                    result.set(&mut cx, i as u32, js_string)?;
                }
                Ok(result.upcast())
            }
        }
    }
}
