#[macro_use]
extern crate neon;
extern crate tokenizers as tk;

use neon::prelude::*;

fn tokenize(mut cx: FunctionContext) -> JsResult<JsArray> {
    let s = cx.argument::<JsString>(0)?.value();
    println!("Tokenizing in rust: {:?}", s);
    let result = tk::tokenize(&s);
    let js_array = JsArray::new(&mut cx, result.len() as u32);
    for (i, token) in result.iter().enumerate() {
        let n = cx.number(*token as f64);
        js_array.set(&mut cx, i as u32, n)?;
    }
    Ok(js_array)
}

register_module!(mut cx, { cx.export_function("tokenize", tokenize) });
