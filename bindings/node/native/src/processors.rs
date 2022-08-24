extern crate tokenizers as tk;

use crate::extraction::*;
use neon::prelude::*;
use std::sync::Arc;

use tk::processors::PostProcessorWrapper;
use tk::Encoding;

/// Processor
#[derive(Clone, Serialize, Deserialize)]
pub struct Processor {
    #[serde(flatten)]
    pub processor: Option<Arc<PostProcessorWrapper>>,
}

impl tk::PostProcessor for Processor {
    fn added_tokens(&self, is_pair: bool) -> usize {
        self.processor
            .as_ref()
            .expect("Uninitialized PostProcessor")
            .added_tokens(is_pair)
    }

    fn process_encodings(
        &self,
        encodings: Vec<Encoding>,
        add_special_tokens: bool,
    ) -> tk::Result<Vec<Encoding>> {
        self.processor
            .as_ref()
            .ok_or("Uninitialized PostProcessor")?
            .process_encodings(encodings, add_special_tokens)
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

/// template_processing(
///   single: String,
///   pair?:  String,
///   special_tokens?: [String, number][] = [],
/// )
fn template_processing(mut cx: FunctionContext) -> JsResult<JsPostProcessor> {
    let mut i = 1;
    let special_tokens = loop {
        if let Ok(Some(spe)) = cx.extract_opt::<Vec<(String, u32)>>(i) {
            break spe;
        }
        i += 1;
        if i == 3 {
            break vec![];
        }
    };

    let single = cx.extract::<String>(0)?;
    let pair = cx.extract_opt::<String>(1)?;

    let mut builder = tk::processors::template::TemplateProcessing::builder();
    builder.try_single(single).map_err(Error)?;
    builder.special_tokens(special_tokens);
    if let Some(pair) = pair {
        builder.try_pair(pair).map_err(Error)?;
    }
    let processor = builder.build().map_err(Error)?;

    let mut js_processor = JsPostProcessor::new::<_, JsPostProcessor, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    js_processor.borrow_mut(&guard).processor = Some(Arc::new(processor.into()));

    Ok(js_processor)
}

/// sequence(processors: List[Processor])
fn sequence(mut cx: FunctionContext) -> JsResult<JsPostProcessor> {
    let processors = cx.argument::<JsArray>(0)?.to_vec(&mut cx)?;
    let mut sequence = Vec::with_capacity(processors.len());

    processors.into_iter().try_for_each(|processor| {
        match processor.downcast::<JsProcessor>().or_throw(&mut cx) {
            Ok(processor) => {
                let guard = cx.lock();
                if let Some(processor_arc) = &processor.borrow(&guard).processor {
                    let processor: ProcessorWrapper = (**processor_arc).clone();
                    sequence.push(processor);
                }
                Ok(())
            }
            Err(e) => Err(e),
        }
    })?;

    let mut pretok = JsProcessor::new::<_, JsProcessor, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    pretok.borrow_mut(&guard).processor = Some(Arc::new(tk::ProcessorWrapper::Sequence(
        tk::processors::sequence::Sequence::new(sequence),
    )));
    Ok(pretok)
}

/// Register everything here
pub fn register(m: &mut ModuleContext, prefix: &str) -> NeonResult<()> {
    m.export_function(&format!("{}_BertProcessing", prefix), bert_processing)?;
    m.export_function(&format!("{}_RobertaProcessing", prefix), roberta_processing)?;
    m.export_function(&format!("{}_ByteLevel", prefix), bytelevel)?;
    m.export_function(
        &format!("{}_TemplateProcessing", prefix),
        template_processing,
    )?;
    m.export_function(&format!("{}_Sequence", prefix), sequence)?;
    Ok(())
}
