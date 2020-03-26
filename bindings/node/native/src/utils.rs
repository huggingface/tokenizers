extern crate tokenizers as tk;

use crate::encoding::JsEncoding;
use neon::prelude::*;

/// slice(s: string, start?: number, end?: number)
fn slice(mut cx: FunctionContext) -> JsResult<JsString> {
    let s = cx.argument::<JsString>(0)?.value();
    let len = s.chars().count();

    let get_index = |x: i32| -> usize {
        if x >= 0 {
            x as usize
        } else {
            (len as i32 + x) as usize
        }
    };

    let begin_index = if let Some(begin_arg) = cx.argument_opt(1) {
        if begin_arg.downcast::<JsUndefined>().is_err() {
            let begin = begin_arg.downcast::<JsNumber>().or_throw(&mut cx)?.value() as i32;
            get_index(begin)
        } else {
            0
        }
    } else {
        0
    };

    let end_index = if let Some(end_arg) = cx.argument_opt(2) {
        if end_arg.downcast::<JsUndefined>().is_err() {
            let end = end_arg.downcast::<JsNumber>().or_throw(&mut cx)?.value() as i32;
            get_index(end)
        } else {
            len
        }
    } else {
        len
    };

    if let Some(slice) = tk::tokenizer::get_range_of(&s, begin_index..end_index) {
        Ok(cx.string(slice))
    } else {
        cx.throw_error("Error in offsets")
    }
}

/// merge_encodings(encodings: Encoding[], growing_offsets: boolean = false): Encoding
fn merge_encodings(mut cx: FunctionContext) -> JsResult<JsEncoding> {
    let encodings = cx
        .argument::<JsArray>(0)?
        .to_vec(&mut cx)?
        .into_iter()
        .map(|item| {
            let encoding = item.downcast::<JsEncoding>().or_throw(&mut cx)?;

            let guard = cx.lock();
            let encoding = encoding
                .borrow(&guard)
                .encoding
                .execute(|e| *e.unwrap().clone());

            Ok(encoding)
        })
        .collect::<Result<Vec<_>, neon::result::Throw>>()
        .map_err(|e| cx.throw_error::<_, ()>(format!("{}", e)).unwrap_err())?;

    let growing_offsets = if let Some(arg) = cx.argument_opt(1) {
        if arg.downcast::<JsUndefined>().is_err() {
            arg.downcast::<JsBoolean>().or_throw(&mut cx)?.value()
        } else {
            false
        }
    } else {
        false
    };

    let new_encoding = tk::tokenizer::Encoding::merge(encodings.as_slice(), growing_offsets);
    let mut js_encoding = JsEncoding::new::<_, JsEncoding, _>(&mut cx, vec![])?;

    let guard = cx.lock();
    js_encoding
        .borrow_mut(&guard)
        .encoding
        .to_owned(Box::new(new_encoding));

    Ok(js_encoding)
}

/// Register everything here
pub fn register(m: &mut ModuleContext, prefix: &str) -> NeonResult<()> {
    m.export_function(&format!("{}_slice", prefix), slice)?;
    m.export_function(&format!("{}_mergeEncodings", prefix), merge_encodings)?;
    Ok(())
}
