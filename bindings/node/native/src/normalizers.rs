extern crate tokenizers as tk;

use crate::container::Container;
use neon::prelude::*;

/// Normalizer
pub struct Normalizer {
    pub normalizer: Container<dyn tk::tokenizer::Normalizer>,
}

declare_types! {
    pub class JsNormalizer for Normalizer {
        init(_) {
            // This should not be called from JS
            Ok(Normalizer {
                normalizer: Container::Empty
            })
        }
    }
}

/// bert_normalizer(options?: {
///   cleanText?: bool = true,
///   handleChineseChars?: bool = true,
///   stripAccents?: bool = true,
///   lowercase?: bool = true
/// })
fn bert_normalizer(mut cx: FunctionContext) -> JsResult<JsNormalizer> {
    let mut clean_text = true;
    let mut handle_chinese_chars = true;
    let mut strip_accents = true;
    let mut lowercase = true;

    if let Some(options) = cx.argument_opt(0) {
        let options = options.downcast::<JsObject>().or_throw(&mut cx)?;
        if let Ok(ct) = options.get(&mut cx, "cleanText") {
            if let Err(_) = ct.downcast::<JsUndefined>() {
                clean_text = ct.downcast::<JsBoolean>().or_throw(&mut cx)?.value();
            }
        }
        if let Ok(hcc) = options.get(&mut cx, "handleChineseChars") {
            if let Err(_) = hcc.downcast::<JsUndefined>() {
                handle_chinese_chars = hcc.downcast::<JsBoolean>().or_throw(&mut cx)?.value();
            }
        }
        if let Ok(sa) = options.get(&mut cx, "stripAccents") {
            if let Err(_) = sa.downcast::<JsUndefined>() {
                strip_accents = sa.downcast::<JsBoolean>().or_throw(&mut cx)?.value();
            }
        }
        if let Ok(l) = options.get(&mut cx, "lowercase") {
            if let Err(_) = l.downcast::<JsUndefined>() {
                lowercase = l.downcast::<JsBoolean>().or_throw(&mut cx)?.value();
            }
        }
    }

    let mut normalizer = JsNormalizer::new::<_, JsNormalizer, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    normalizer.borrow_mut(&guard).normalizer.to_owned(Box::new(
        tk::normalizers::bert::BertNormalizer::new(
            clean_text,
            handle_chinese_chars,
            strip_accents,
            lowercase,
        ),
    ));
    Ok(normalizer)
}

/// nfd()
fn nfd(mut cx: FunctionContext) -> JsResult<JsNormalizer> {
    let mut normalizer = JsNormalizer::new::<_, JsNormalizer, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    normalizer
        .borrow_mut(&guard)
        .normalizer
        .to_owned(Box::new(tk::normalizers::unicode::NFD));
    Ok(normalizer)
}

/// nfkd()
fn nfkd(mut cx: FunctionContext) -> JsResult<JsNormalizer> {
    let mut normalizer = JsNormalizer::new::<_, JsNormalizer, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    normalizer
        .borrow_mut(&guard)
        .normalizer
        .to_owned(Box::new(tk::normalizers::unicode::NFKD));
    Ok(normalizer)
}

/// nfc()
fn nfc(mut cx: FunctionContext) -> JsResult<JsNormalizer> {
    let mut normalizer = JsNormalizer::new::<_, JsNormalizer, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    normalizer
        .borrow_mut(&guard)
        .normalizer
        .to_owned(Box::new(tk::normalizers::unicode::NFC));
    Ok(normalizer)
}

/// nfkc()
fn nfkc(mut cx: FunctionContext) -> JsResult<JsNormalizer> {
    let mut normalizer = JsNormalizer::new::<_, JsNormalizer, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    normalizer
        .borrow_mut(&guard)
        .normalizer
        .to_owned(Box::new(tk::normalizers::unicode::NFKC));
    Ok(normalizer)
}

/// strip(left?: boolean, right?: boolean)
fn strip(mut cx: FunctionContext) -> JsResult<JsNormalizer> {
    let mut left = true;
    let mut right = true;

    if let Some(left_arg) = cx.argument_opt(0) {
        if left_arg.downcast::<JsUndefined>().is_err() {
            left = left_arg.downcast_or_throw::<JsBoolean, _>(&mut cx)?.value();
        }

        if let Some(right_arg) = cx.argument_opt(1) {
            if right_arg.downcast::<JsUndefined>().is_err() {
                right = right_arg
                    .downcast_or_throw::<JsBoolean, _>(&mut cx)?
                    .value();
            }
        }
    }

    let mut normalizer = JsNormalizer::new::<_, JsNormalizer, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    normalizer
        .borrow_mut(&guard)
        .normalizer
        .to_owned(Box::new(tk::normalizers::strip::Strip::new(left, right)));

    Ok(normalizer)
}

/// sequence(normalizers: Normalizer[])
fn sequence(mut cx: FunctionContext) -> JsResult<JsNormalizer> {
    let normalizers = cx
        .argument::<JsArray>(0)?
        .to_vec(&mut cx)?
        .into_iter()
        .map(|normalizer| {
            match normalizer.downcast::<JsNormalizer>().or_throw(&mut cx) {
                Ok(normalizer) => {
                     let guard = cx.lock();
                     if !normalizer.borrow(&guard).normalizer.is_owned() {
                         cx.throw_error("At least one of the normalizers is already being used in another Tokenizer")
                     } else {
                         Ok(normalizer)
                     }
                },
                Err(e) => Err(e)
            }
        })
        .collect::<NeonResult<Vec<_>>>()?;

    // We've checked that all the normalizers can be used, now we can convert them and attach
    // them the to sequence normalizer
    let normalizers = normalizers
        .into_iter()
        .map(|mut normalizer| {
            let guard = cx.lock();
            let n = normalizer
                .borrow_mut(&guard)
                .normalizer
                .to_pointer()
                .unwrap();
            n
        })
        .collect::<Vec<_>>();

    let mut normalizer = JsNormalizer::new::<_, JsNormalizer, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    normalizer
        .borrow_mut(&guard)
        .normalizer
        .to_owned(Box::new(tk::normalizers::utils::Sequence::new(normalizers)));
    Ok(normalizer)
}

/// lowercase()
fn lowercase(mut cx: FunctionContext) -> JsResult<JsNormalizer> {
    let mut normalizer = JsNormalizer::new::<_, JsNormalizer, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    normalizer
        .borrow_mut(&guard)
        .normalizer
        .to_owned(Box::new(tk::normalizers::utils::Lowercase));
    Ok(normalizer)
}

/// Register everything here
pub fn register(m: &mut ModuleContext, prefix: &str) -> NeonResult<()> {
    m.export_function(&format!("{}_BertNormalizer", prefix), bert_normalizer)?;
    m.export_function(&format!("{}_NFD", prefix), nfd)?;
    m.export_function(&format!("{}_NFKD", prefix), nfkd)?;
    m.export_function(&format!("{}_NFC", prefix), nfc)?;
    m.export_function(&format!("{}_NFKC", prefix), nfkc)?;
    m.export_function(&format!("{}_Sequence", prefix), sequence)?;
    m.export_function(&format!("{}_Lowercase", prefix), lowercase)?;
    m.export_function(&format!("{}_Strip", prefix), strip)?;
    Ok(())
}
