extern crate tokenizers as tk;

use crate::extraction::*;
use neon::prelude::*;
use std::sync::Arc;

use tk::pre_tokenizers::PreTokenizerWrapper;
use tk::PreTokenizedString;

/// PreTokenizers
#[derive(Clone)]
pub struct PreTokenizer {
    pub pretok: Option<Arc<PreTokenizerWrapper>>,
}

impl tk::PreTokenizer for PreTokenizer {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> tk::Result<()> {
        self.pretok
            .as_ref()
            .ok_or("Uninitialized PreTokenizer")?
            .pre_tokenize(pretokenized)
    }
}

declare_types! {
    pub class JsPreTokenizer for PreTokenizer {
        init(_) {
            // This should not be called from JS
            Ok(PreTokenizer { pretok: None })
        }
    }
}

/// byte_level(addPrefixSpace: bool = true)
fn byte_level(mut cx: FunctionContext) -> JsResult<JsPreTokenizer> {
    let mut byte_level = tk::pre_tokenizers::byte_level::ByteLevel::default();
    if let Some(add_prefix_space) = cx.extract_opt::<bool>(0)? {
        byte_level = byte_level.add_prefix_space(add_prefix_space);
    }

    let mut pretok = JsPreTokenizer::new::<_, JsPreTokenizer, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    pretok.borrow_mut(&guard).pretok = Some(Arc::new(byte_level.into()));
    Ok(pretok)
}

/// byte_level_alphabet()
fn byte_level_alphabet(mut cx: FunctionContext) -> JsResult<JsValue> {
    let chars = tk::pre_tokenizers::byte_level::ByteLevel::alphabet()
        .into_iter()
        .map(|c| c.to_string())
        .collect::<Vec<_>>();

    Ok(neon_serde::to_value(&mut cx, &chars)?)
}

/// whitespace()
fn whitespace(mut cx: FunctionContext) -> JsResult<JsPreTokenizer> {
    let mut pretok = JsPreTokenizer::new::<_, JsPreTokenizer, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    pretok.borrow_mut(&guard).pretok = Some(Arc::new(
        tk::pre_tokenizers::whitespace::Whitespace::default().into(),
    ));
    Ok(pretok)
}

/// whitespace_split()
fn whitespace_split(mut cx: FunctionContext) -> JsResult<JsPreTokenizer> {
    let mut pretok = JsPreTokenizer::new::<_, JsPreTokenizer, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    pretok.borrow_mut(&guard).pretok = Some(Arc::new(
        tk::pre_tokenizers::whitespace::WhitespaceSplit.into(),
    ));
    Ok(pretok)
}

/// bert_pre_tokenizer()
fn bert_pre_tokenizer(mut cx: FunctionContext) -> JsResult<JsPreTokenizer> {
    let mut pretok = JsPreTokenizer::new::<_, JsPreTokenizer, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    pretok.borrow_mut(&guard).pretok =
        Some(Arc::new(tk::pre_tokenizers::bert::BertPreTokenizer.into()));
    Ok(pretok)
}

/// metaspace(replacement: string = '_', addPrefixSpace: bool = true)
fn metaspace(mut cx: FunctionContext) -> JsResult<JsPreTokenizer> {
    let replacement = cx.extract_opt::<char>(0)?.unwrap_or('▁');
    let add_prefix_space = cx.extract_opt::<bool>(1)?.unwrap_or(true);

    let mut pretok = JsPreTokenizer::new::<_, JsPreTokenizer, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    pretok.borrow_mut(&guard).pretok = Some(Arc::new(
        tk::pre_tokenizers::metaspace::Metaspace::new(replacement, add_prefix_space).into(),
    ));
    Ok(pretok)
}

/// char_delimiter_split(delimiter: string)
fn char_delimiter_split(mut cx: FunctionContext) -> JsResult<JsPreTokenizer> {
    let delimiter = cx.extract::<char>(0)?;

    let mut pretok = JsPreTokenizer::new::<_, JsPreTokenizer, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    pretok.borrow_mut(&guard).pretok = Some(Arc::new(
        tk::pre_tokenizers::delimiter::CharDelimiterSplit::new(delimiter).into(),
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
