extern crate tokenizers as tk;

use crate::extraction::*;
use neon::prelude::*;
use serde::{ser::SerializeStruct, Serialize, Serializer};
use std::sync::Arc;

use tk::normalizers::NormalizerWrapper;
use tk::NormalizedString;

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum JsNormalizerWrapper {
    Sequence(Vec<Arc<NormalizerWrapper>>),
    Wrapped(Arc<NormalizerWrapper>),
}

impl Serialize for JsNormalizerWrapper {
    fn serialize<S>(&self, serializer: S) -> Result<<S as Serializer>::Ok, <S as Serializer>::Error>
    where
        S: Serializer,
    {
        match self {
            JsNormalizerWrapper::Sequence(seq) => {
                let mut ser = serializer.serialize_struct("Sequence", 2)?;
                ser.serialize_field("type", "Sequence")?;
                ser.serialize_field("normalizers", seq)?;
                ser.end()
            }
            JsNormalizerWrapper::Wrapped(inner) => inner.serialize(serializer),
        }
    }
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Normalizer {
    #[serde(flatten)]
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

        method normalizeString(mut cx) {
            use tk::Normalizer;

            let sequence = cx.extract::<String>(0)?;
            let mut normalized = NormalizedString::from(sequence);

            let this = cx.this();
            let guard = cx.lock();
            this.borrow(&guard)
                .normalize(&mut normalized)
                .map_err(|e| Error(format!("{}", e)))?;

            Ok(cx.string(normalized.get()).upcast())
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
/// strip_accents()
fn strip_accents(mut cx: FunctionContext) -> JsResult<JsNormalizer> {
    let mut normalizer = JsNormalizer::new::<_, JsNormalizer, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    normalizer.borrow_mut(&guard).normalizer = Some(tk::normalizers::strip::StripAccents.into());

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
                            JsNormalizerWrapper::Sequence(seq) => sequence.extend(seq),
                            JsNormalizerWrapper::Wrapped(inner) => sequence.push(inner),
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
    normalizer.borrow_mut(&guard).normalizer = Some(JsNormalizerWrapper::Sequence(sequence));
    Ok(normalizer)
}

/// lowercase()
fn lowercase(mut cx: FunctionContext) -> JsResult<JsNormalizer> {
    let mut normalizer = JsNormalizer::new::<_, JsNormalizer, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    normalizer.borrow_mut(&guard).normalizer = Some(tk::normalizers::utils::Lowercase.into());
    Ok(normalizer)
}

/// replace()
fn replace(mut cx: FunctionContext) -> JsResult<JsNormalizer> {
    let pattern: String = cx.extract::<String>(0)?;
    let content: String = cx.extract::<String>(1)?;
    let mut normalizer = JsNormalizer::new::<_, JsNormalizer, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    normalizer.borrow_mut(&guard).normalizer = Some(
        tk::normalizers::replace::Replace::new(pattern, content)
            .map_err(|e| Error(e.to_string()))?
            .into(),
    );
    Ok(normalizer)
}

/// nmt()
fn nmt(mut cx: FunctionContext) -> JsResult<JsNormalizer> {
    let mut normalizer = JsNormalizer::new::<_, JsNormalizer, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    normalizer.borrow_mut(&guard).normalizer = Some(tk::normalizers::unicode::Nmt.into());
    Ok(normalizer)
}

/// precompiled()
fn precompiled(mut cx: FunctionContext) -> JsResult<JsNormalizer> {
    let bytes = cx.extract::<Vec<u8>>(0)?;
    let mut normalizer = JsNormalizer::new::<_, JsNormalizer, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    normalizer.borrow_mut(&guard).normalizer = Some(
        tk::normalizers::precompiled::Precompiled::from(&bytes)
            .map_err(|e| Error(e.to_string()))?
            .into(),
    );
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
    m.export_function(&format!("{}_StripAccents", prefix), strip_accents)?;
    m.export_function(&format!("{}_Nmt", prefix), nmt)?;
    m.export_function(&format!("{}_Precompiled", prefix), precompiled)?;
    m.export_function(&format!("{}_Replace", prefix), replace)?;
    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;
    use tk::normalizers::unicode::{NFC, NFKC};
    use tk::normalizers::utils::Sequence;
    use tk::normalizers::NormalizerWrapper;

    #[test]
    fn serialize() {
        let js_wrapped: JsNormalizerWrapper = NFKC.into();
        let js_ser = serde_json::to_string(&js_wrapped).unwrap();

        let rs_wrapped = NormalizerWrapper::NFKC(NFKC);
        let rs_ser = serde_json::to_string(&rs_wrapped).unwrap();
        assert_eq!(js_ser, rs_ser);

        let js_norm: Normalizer = serde_json::from_str(&rs_ser).unwrap();
        match js_norm.normalizer.unwrap() {
            JsNormalizerWrapper::Wrapped(nfc) => match nfc.as_ref() {
                NormalizerWrapper::NFKC(_) => {}
                _ => panic!("Expected NFKC"),
            },
            _ => panic!("Expected wrapped, not sequence."),
        }

        let js_seq: JsNormalizerWrapper = Sequence::new(vec![NFC.into(), NFKC.into()]).into();
        let js_wrapper_ser = serde_json::to_string(&js_seq).unwrap();
        let rs_wrapped = NormalizerWrapper::Sequence(Sequence::new(vec![NFC.into(), NFKC.into()]));
        let rs_ser = serde_json::to_string(&rs_wrapped).unwrap();
        assert_eq!(js_wrapper_ser, rs_ser);

        let js_seq = Normalizer {
            normalizer: Some(js_seq),
        };
        let js_ser = serde_json::to_string(&js_seq).unwrap();
        assert_eq!(js_wrapper_ser, js_ser);

        let rs_seq = Sequence::new(vec![NFC.into(), NFKC.into()]);
        let rs_ser = serde_json::to_string(&rs_seq).unwrap();
        assert_eq!(js_wrapper_ser, rs_ser);
    }
}
