extern crate tokenizers as tk;

use crate::encoding::JsEncoding;
use crate::extraction::*;
use crate::tokenizer::Encoding;
use neon::prelude::*;

/// slice(s: string, start?: number, end?: number)
fn slice(mut cx: FunctionContext) -> JsResult<JsString> {
    let s = cx.extract::<String>(0)?;
    let len = s.chars().count();

    let get_index = |x: i32| -> usize {
        if x >= 0 {
            x as usize
        } else {
            (len as i32 + x) as usize
        }
    };

    let begin_index = get_index(cx.extract_opt::<i32>(1)?.unwrap_or(0));
    let end_index = get_index(cx.extract_opt::<i32>(2)?.unwrap_or(len as i32));

    if let Some(slice) = tk::tokenizer::normalizer::get_range_of(&s, begin_index..end_index) {
        Ok(cx.string(slice))
    } else {
        cx.throw_error("Error in offsets")
    }
}

/// merge_encodings(encodings: Encoding[], growing_offsets: boolean = false): Encoding
fn merge_encodings(mut cx: FunctionContext) -> JsResult<JsEncoding> {
    let encodings: Vec<tk::Encoding> = cx
        .extract_vec::<Encoding>(0)?
        .into_iter()
        .map(|e| e.into())
        .collect();
    let growing_offsets = cx.extract_opt::<bool>(1)?.unwrap_or(false);

    let new_encoding = tk::tokenizer::Encoding::merge(encodings, growing_offsets);
    let mut js_encoding = JsEncoding::new::<_, JsEncoding, _>(&mut cx, vec![])?;

    let guard = cx.lock();
    js_encoding.borrow_mut(&guard).encoding = Some(new_encoding);

    Ok(js_encoding)
}

/// Register everything here
pub fn register(m: &mut ModuleContext, prefix: &str) -> NeonResult<()> {
    m.export_function(&format!("{}_slice", prefix), slice)?;
    m.export_function(&format!("{}_mergeEncodings", prefix), merge_encodings)?;
    Ok(())
}
