extern crate tokenizers as tk;

use crate::extraction::*;
use neon::prelude::*;
use std::sync::Arc;

use tk::processors::PostProcessorWrapper;
use tk::Encoding;

/// Processor
#[derive(Clone)]
pub struct PostProcessor {
    pub processor: Option<Arc<PostProcessorWrapper>>,
}

impl tk::PostProcessor for Processor {
    fn added_tokens(&self, is_pair: bool) -> usize {
        self.processor
            .as_ref()
            .expect("Uninitialized PostProcessor")
            .added_tokens(is_pair)
    }

    fn process(
        &self,
        encoding: Encoding,
        pair_encoding: Option<Encoding>,
        add_special_tokens: bool,
    ) -> tk::Result<Encoding> {
        self.processor
            .as_ref()
            .ok_or("Uninitialized PostProcessor")?
            .process(encoding, pair_encoding, add_special_tokens)
    }
}

declare_types! {
    pub class JsPostProcessor for Processor {
        init(_) {
            // This should not be called from JS
            Ok(Processor { processor: None })
        }
    }
}

/// bert_processing(sep: [String, number], cls: [String, number])
fn bert_processing(mut cx: FunctionContext) -> JsResult<JsPostProcessor> {
    let sep = cx.extract::<(String, u32)>(0)?;
    let cls = cx.extract::<(String, u32)>(1)?;

    let mut processor = JsPostProcessor::new::<_, JsPostProcessor, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    processor.borrow_mut(&guard).processor = Some(Arc::new(
        tk::processors::bert::BertProcessing::new(sep, cls).into(),
    ));
    Ok(processor)
}

/// roberta_processing(
///   sep: [String, number],
///   cls: [String, number],
///   trimOffsets: boolean = true,
///   addPrefixSpace: boolean = true
/// )
fn roberta_processing(mut cx: FunctionContext) -> JsResult<JsPostProcessor> {
    let sep = cx.extract::<(String, u32)>(0)?;
    let cls = cx.extract::<(String, u32)>(1)?;

    let mut processor = tk::processors::roberta::RobertaProcessing::new(sep, cls);
    if let Some(trim_offsets) = cx.extract_opt::<bool>(2)? {
        processor = processor.trim_offsets(trim_offsets);
    }
    if let Some(add_prefix_space) = cx.extract_opt::<bool>(3)? {
        processor = processor.add_prefix_space(add_prefix_space);
    }

    let mut js_processor = JsPostProcessor::new::<_, JsPostProcessor, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    js_processor.borrow_mut(&guard).processor = Some(Arc::new(processor.into()));
    Ok(js_processor)
}

/// bytelevel(trimOffsets?: boolean)
fn bytelevel(mut cx: FunctionContext) -> JsResult<JsPostProcessor> {
    let mut byte_level = tk::processors::byte_level::ByteLevel::default();

    if let Some(trim_offsets) = cx.extract_opt::<bool>(0)? {
        byte_level = byte_level.trim_offsets(trim_offsets);
    }

    let mut processor = JsPostProcessor::new::<_, JsPostProcessor, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    processor.borrow_mut(&guard).processor = Some(Arc::new(byte_level.into()));
    Ok(processor)
}

/// Register everything here
pub fn register(m: &mut ModuleContext, prefix: &str) -> NeonResult<()> {
    m.export_function(&format!("{}_BertProcessing", prefix), bert_processing)?;
    m.export_function(&format!("{}_RobertaProcessing", prefix), roberta_processing)?;
    m.export_function(&format!("{}_ByteLevel", prefix), bytelevel)?;
    Ok(())
}
