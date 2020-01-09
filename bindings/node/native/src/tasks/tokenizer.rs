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
    Single(WorkingTokenizer, Option<EncodeInput>),
    Batch(WorkingTokenizer, Option<Vec<EncodeInput>>),
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
            EncodeTask::Single(worker, input) => {
                let mut input = unsafe { std::ptr::read(input) };
                let tokenizer: &Tokenizer = unsafe { &*worker.ptr };
                tokenizer
                    .encode(input.take().ok_or("No provided input")?)
                    .map_err(|e| format!("{}", e))
                    .map(|encoding| EncodeOutput::Single(encoding))
            }
            EncodeTask::Batch(worker, input) => {
                let mut input = unsafe { std::ptr::read(input) };
                let tokenizer: &Tokenizer = unsafe { &*worker.ptr };
                tokenizer
                    .encode_batch(input.take().ok_or("No provided input")?)
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
