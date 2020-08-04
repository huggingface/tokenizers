extern crate tokenizers as tk;

use crate::extraction::*;
use neon::prelude::*;
use std::sync::Arc;

use tk::normalizers::NormalizerWrapper;
use tk::NormalizedString;

#[derive(Clone)]
pub enum JsNormalizerWrapper {
    Sequence(Vec<Arc<NormalizerWrapper>>),
    Wrapped(Arc<NormalizerWrapper>),
}

impl<I> From<I> for JsNormalizerWrapper
where
    I: Into<NormalizerWrapper>,
{
    fn from(norm: I) -> Self {
        JsNormalizerWrapper::Wrapped(Arc::new(norm.into()))
    }
}

/// Normalizer
#[derive(Clone)]
pub struct Normalizer {
    pub normalizer: Option<JsNormalizerWrapper>,
}

impl tk::Normalizer for Normalizer {
    fn normalize(&self, normalized: &mut NormalizedString) -> tk::Result<()> {
        match self.normalizer.as_ref().ok_or("Uninitialized Normalizer")? {
            JsNormalizerWrapper::Sequence(seq) => {
                for norm in seq {
                    norm.normalize(normalized)?;
                }
            }
            JsNormalizerWrapper::Wrapped(norm) => norm.normalize(normalized)?,
        };

        Ok(())
    }
}

declare_types! {
    pub class JsNormalizer for Normalizer {
        init(_) {
            // This should not be called from JS
            Ok(Normalizer { normalizer: None })
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
    normalizer.borrow_mut(&guard).normalizer = Some(
        tk::normalizers::bert::BertNormalizer::new(
            options.clean_text,
            options.handle_chinese_chars,
            options.strip_accents,
            options.lowercase,
        )
        .into(),
    );
    Ok(normalizer)
}

/// nfd()
fn nfd(mut cx: FunctionContext) -> JsResult<JsNormalizer> {
    let mut normalizer = JsNormalizer::new::<_, JsNormalizer, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    normalizer.borrow_mut(&guard).normalizer = Some(tk::normalizers::unicode::NFD.into());
    Ok(normalizer)
}

/// nfkd()
fn nfkd(mut cx: FunctionContext) -> JsResult<JsNormalizer> {
    let mut normalizer = JsNormalizer::new::<_, JsNormalizer, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    normalizer.borrow_mut(&guard).normalizer = Some(tk::normalizers::unicode::NFKD.into());
    Ok(normalizer)
}

/// nfc()
fn nfc(mut cx: FunctionContext) -> JsResult<JsNormalizer> {
    let mut normalizer = JsNormalizer::new::<_, JsNormalizer, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    normalizer.borrow_mut(&guard).normalizer = Some(tk::normalizers::unicode::NFC.into());
    Ok(normalizer)
}

/// nfkc()
fn nfkc(mut cx: FunctionContext) -> JsResult<JsNormalizer> {
    let mut normalizer = JsNormalizer::new::<_, JsNormalizer, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    normalizer.borrow_mut(&guard).normalizer = Some(tk::normalizers::unicode::NFKC.into());
    Ok(normalizer)
}

/// strip(left?: boolean, right?: boolean)
fn strip(mut cx: FunctionContext) -> JsResult<JsNormalizer> {
    let left = cx.extract_opt::<bool>(0)?.unwrap_or(true);
    let right = cx.extract_opt::<bool>(1)?.unwrap_or(true);

    let mut normalizer = JsNormalizer::new::<_, JsNormalizer, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    normalizer.borrow_mut(&guard).normalizer =
        Some(tk::normalizers::strip::Strip::new(left, right).into());

    Ok(normalizer)
}

/// sequence(normalizers: Normalizer[])
fn sequence(mut cx: FunctionContext) -> JsResult<JsNormalizer> {
    let normalizers = cx.argument::<JsArray>(0)?.to_vec(&mut cx)?;
    let mut sequence = Vec::with_capacity(normalizers.len());

    normalizers
        .into_iter()
        .map(
            |normalizer| match normalizer.downcast::<JsNormalizer>().or_throw(&mut cx) {
                Ok(normalizer) => {
                    let guard = cx.lock();
                    let normalizer = normalizer.borrow(&guard).normalizer.clone();
                    if let Some(normalizer) = normalizer {
                        match normalizer {
                            JsNormalizerWrapper::Sequence(seq) => {
                                sequence.extend(seq.iter().map(|i| i.clone()));
                            }
                            JsNormalizerWrapper::Wrapped(inner) => sequence.push(inner.clone()),
                        }
                        Ok(())
                    } else {
                        cx.throw_error("Uninitialized Normalizer")
                    }
                }
                Err(e) => Err(e),
            },
        )
        .collect::<NeonResult<_>>()?;

    let mut normalizer = JsNormalizer::new::<_, JsNormalizer, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    normalizer.borrow_mut(&guard).normalizer = Some(JsNormalizerWrapper::Sequence(sequence).into());
    Ok(normalizer)
}

/// lowercase()
fn lowercase(mut cx: FunctionContext) -> JsResult<JsNormalizer> {
    let mut normalizer = JsNormalizer::new::<_, JsNormalizer, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    normalizer.borrow_mut(&guard).normalizer = Some(tk::normalizers::utils::Lowercase.into());
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
