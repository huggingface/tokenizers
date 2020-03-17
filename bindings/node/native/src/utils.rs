extern crate tokenizers as tk;

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

/// Register everything here
pub fn register(m: &mut ModuleContext, prefix: &str) -> NeonResult<()> {
    m.export_function(&format!("{}_slice", prefix), slice)?;
    Ok(())
}
