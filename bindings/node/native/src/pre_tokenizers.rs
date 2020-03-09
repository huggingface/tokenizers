extern crate tokenizers as tk;

use crate::utils::Container;
use neon::prelude::*;

/// PreTokenizers
pub struct PreTokenizer {
    pub pretok: Container<dyn tk::tokenizer::PreTokenizer + Sync>,
}

declare_types! {
    pub class JsPreTokenizer for PreTokenizer {
        init(_) {
            // This should not be called from JS
            Ok(PreTokenizer {
                pretok: Container::Empty
            })
        }
    }
}

/// byte_level(addPrefixSpace: bool = true)
fn byte_level(mut cx: FunctionContext) -> JsResult<JsPreTokenizer> {
    let mut add_prefix_space = true;

    if let Some(args) = cx.argument_opt(0) {
        if args.downcast::<JsUndefined>().is_err() {
            add_prefix_space = args.downcast::<JsBoolean>().or_throw(&mut cx)?.value();
        }
    }

    let mut pretok = JsPreTokenizer::new::<_, JsPreTokenizer, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    pretok.borrow_mut(&guard).pretok.to_owned(Box::new(
        tk::pre_tokenizers::byte_level::ByteLevel::new(add_prefix_space),
    ));
    Ok(pretok)
}

/// byte_level_alphabet()
fn byte_level_alphabet(mut cx: FunctionContext) -> JsResult<JsArray> {
    let chars = tk::pre_tokenizers::byte_level::ByteLevel::alphabet()
        .into_iter()
        .map(|c| c.to_string())
        .collect::<Vec<_>>();

    let js_chars = JsArray::new(&mut cx, chars.len() as u32);
    for (i, c) in chars.into_iter().enumerate() {
        let s = cx.string(c);
        js_chars.set(&mut cx, i as u32, s)?;
    }

    Ok(js_chars)
}

/// whitespace()
fn whitespace(mut cx: FunctionContext) -> JsResult<JsPreTokenizer> {
    let mut pretok = JsPreTokenizer::new::<_, JsPreTokenizer, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    pretok
        .borrow_mut(&guard)
        .pretok
        .to_owned(Box::new(tk::pre_tokenizers::whitespace::Whitespace));
    Ok(pretok)
}

/// whitespace_split()
fn whitespace_split(mut cx: FunctionContext) -> JsResult<JsPreTokenizer> {
    let mut pretok = JsPreTokenizer::new::<_, JsPreTokenizer, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    pretok
        .borrow_mut(&guard)
        .pretok
        .to_owned(Box::new(tk::pre_tokenizers::whitespace::WhitespaceSplit));
    Ok(pretok)
}

/// bert_pre_tokenizer()
fn bert_pre_tokenizer(mut cx: FunctionContext) -> JsResult<JsPreTokenizer> {
    let mut pretok = JsPreTokenizer::new::<_, JsPreTokenizer, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    pretok
        .borrow_mut(&guard)
        .pretok
        .to_owned(Box::new(tk::pre_tokenizers::bert::BertPreTokenizer));
    Ok(pretok)
}

/// metaspace(replacement: string = '_', addPrefixSpace: bool = true)
fn metaspace(mut cx: FunctionContext) -> JsResult<JsPreTokenizer> {
    let mut replacement = '▁';
    if let Some(args) = cx.argument_opt(0) {
        if args.downcast::<JsUndefined>().is_err() {
            let rep = args.downcast::<JsString>().or_throw(&mut cx)?.value() as String;
            replacement = rep.chars().nth(0).ok_or_else(|| {
                cx.throw_error::<_, ()>("replacement must be a character")
                    .unwrap_err()
            })?;
        }
    };

    let mut add_prefix_space = true;
    if let Some(args) = cx.argument_opt(1) {
        if args.downcast::<JsUndefined>().is_err() {
            add_prefix_space = args.downcast::<JsBoolean>().or_throw(&mut cx)?.value() as bool;
        }
    }

    let mut pretok = JsPreTokenizer::new::<_, JsPreTokenizer, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    pretok.borrow_mut(&guard).pretok.to_owned(Box::new(
        tk::pre_tokenizers::metaspace::Metaspace::new(replacement, add_prefix_space),
    ));
    Ok(pretok)
}

/// char_delimiter_split(delimiter: string)
fn char_delimiter_split(mut cx: FunctionContext) -> JsResult<JsPreTokenizer> {
    let argument = cx.argument::<JsString>(0)?.value();
    let delimiter = argument.chars().nth(0).ok_or_else(|| {
        cx.throw_error::<_, ()>("delimiter must be a character")
            .unwrap_err()
    })?;

    let mut pretok = JsPreTokenizer::new::<_, JsPreTokenizer, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    pretok.borrow_mut(&guard).pretok.to_owned(Box::new(
        tk::pre_tokenizers::delimiter::CharDelimiterSplit::new(delimiter),
    ));

    Ok(pretok)
}

/// Register everything here
pub fn register(m: &mut ModuleContext, prefix: &str) -> NeonResult<()> {
    m.export_function(&format!("{}_ByteLevel", prefix), byte_level)?;
    m.export_function(
        &format!("{}_ByteLevel_Alphabet", prefix),
        byte_level_alphabet,
    )?;
    m.export_function(&format!("{}_Whitespace", prefix), whitespace)?;
    m.export_function(&format!("{}_WhitespaceSplit", prefix), whitespace_split)?;
    m.export_function(&format!("{}_BertPreTokenizer", prefix), bert_pre_tokenizer)?;
    m.export_function(&format!("{}_Metaspace", prefix), metaspace)?;
    m.export_function(
        &format!("{}_CharDelimiterSplit", prefix),
        char_delimiter_split,
    )?;
    Ok(())
}
