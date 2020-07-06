extern crate tokenizers as tk;

use crate::container::Container;
use crate::extraction::*;
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

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BertNormalizerOptions {
    clean_text: bool,
    handle_chinese_chars: bool,
    strip_accents: Option<bool>,
    lowercase: bool,
}
impl Default for BertNormalizerOptions {
    fn default() -> Self {
        Self {
            clean_text: true,
            handle_chinese_chars: true,
            strip_accents: None,
            lowercase: true,
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
    let options = cx
        .extract_opt::<BertNormalizerOptions>(0)?
        .unwrap_or_else(BertNormalizerOptions::default);

    let mut normalizer = JsNormalizer::new::<_, JsNormalizer, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    normalizer
        .borrow_mut(&guard)
        .normalizer
        .make_owned(Box::new(tk::normalizers::bert::BertNormalizer::new(
            options.clean_text,
            options.handle_chinese_chars,
            options.strip_accents,
            options.lowercase,
        )));
    Ok(normalizer)
}

/// nfd()
fn nfd(mut cx: FunctionContext) -> JsResult<JsNormalizer> {
    let mut normalizer = JsNormalizer::new::<_, JsNormalizer, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    normalizer
        .borrow_mut(&guard)
        .normalizer
        .make_owned(Box::new(tk::normalizers::unicode::NFD));
    Ok(normalizer)
}

/// nfkd()
fn nfkd(mut cx: FunctionContext) -> JsResult<JsNormalizer> {
    let mut normalizer = JsNormalizer::new::<_, JsNormalizer, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    normalizer
        .borrow_mut(&guard)
        .normalizer
        .make_owned(Box::new(tk::normalizers::unicode::NFKD));
    Ok(normalizer)
}

/// nfc()
fn nfc(mut cx: FunctionContext) -> JsResult<JsNormalizer> {
    let mut normalizer = JsNormalizer::new::<_, JsNormalizer, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    normalizer
        .borrow_mut(&guard)
        .normalizer
        .make_owned(Box::new(tk::normalizers::unicode::NFC));
    Ok(normalizer)
}

/// nfkc()
fn nfkc(mut cx: FunctionContext) -> JsResult<JsNormalizer> {
    let mut normalizer = JsNormalizer::new::<_, JsNormalizer, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    normalizer
        .borrow_mut(&guard)
        .normalizer
        .make_owned(Box::new(tk::normalizers::unicode::NFKC));
    Ok(normalizer)
}

/// strip(left?: boolean, right?: boolean)
fn strip(mut cx: FunctionContext) -> JsResult<JsNormalizer> {
    let left = cx.extract_opt::<bool>(0)?.unwrap_or(true);
    let right = cx.extract_opt::<bool>(1)?.unwrap_or(true);

    let mut normalizer = JsNormalizer::new::<_, JsNormalizer, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    normalizer
        .borrow_mut(&guard)
        .normalizer
        .make_owned(Box::new(tk::normalizers::strip::Strip::new(left, right)));

    Ok(normalizer)
}

/// sequence(normalizers: Normalizer[])
fn sequence(mut cx: FunctionContext) -> JsResult<JsNormalizer> {
    let normalizers = cx
        .argument::<JsArray>(0)?
        .to_vec(&mut cx)?
        .into_iter()
        .map(
            |normalizer| match normalizer.downcast::<JsNormalizer>().or_throw(&mut cx) {
                Ok(normalizer) => {
                    let guard = cx.lock();
                    if !normalizer.borrow(&guard).normalizer.is_owned() {
                        cx.throw_error(
                            "At least one of the normalizers is \
                                        already being used in another Tokenizer",
                        )
                    } else {
                        Ok(normalizer)
                    }
                }
                Err(e) => Err(e),
            },
        )
        .collect::<NeonResult<Vec<_>>>()?;

    // We've checked that all the normalizers can be used, now we can convert them and attach
    // them the to sequence normalizer
    let normalizers = normalizers
        .into_iter()
        .map(|mut normalizer| {
            let guard = cx.lock();
            let mut n = normalizer.borrow_mut(&guard);
            n.normalizer.make_pointer().unwrap()
        })
        .collect::<Vec<_>>();

    let mut normalizer = JsNormalizer::new::<_, JsNormalizer, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    normalizer
        .borrow_mut(&guard)
        .normalizer
        .make_owned(Box::new(tk::normalizers::utils::Sequence::new(normalizers)));
    Ok(normalizer)
}

/// lowercase()
fn lowercase(mut cx: FunctionContext) -> JsResult<JsNormalizer> {
    let mut normalizer = JsNormalizer::new::<_, JsNormalizer, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    normalizer
        .borrow_mut(&guard)
        .normalizer
        .make_owned(Box::new(tk::normalizers::utils::Lowercase));
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
