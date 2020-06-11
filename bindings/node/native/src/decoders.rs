extern crate tokenizers as tk;

use crate::container::Container;
use crate::extraction::*;
use neon::prelude::*;

/// Decoder
pub struct Decoder {
    pub decoder: Container<dyn tk::tokenizer::Decoder>,
}

declare_types! {
    pub class JsDecoder for Decoder {
        init(_) {
            // This should not be called from JS
            Ok(Decoder {
                decoder: Container::Empty
            })
        }
    }
}

/// byte_level()
fn byte_level(mut cx: FunctionContext) -> JsResult<JsDecoder> {
    let mut decoder = JsDecoder::new::<_, JsDecoder, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    decoder
        .borrow_mut(&guard)
        .decoder
        .make_owned(Box::new(tk::decoders::byte_level::ByteLevel::default()));
    Ok(decoder)
}

/// wordpiece(prefix: String = "##", cleanup: bool)
fn wordpiece(mut cx: FunctionContext) -> JsResult<JsDecoder> {
    let prefix = cx
        .extract_opt::<String>(0)?
        .unwrap_or_else(|| String::from("##"));
    let cleanup = cx.extract_opt::<bool>(1)?.unwrap_or(true);

    let mut decoder = JsDecoder::new::<_, JsDecoder, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    decoder.borrow_mut(&guard).decoder.make_owned(Box::new(
        tk::decoders::wordpiece::WordPiece::new(prefix, cleanup),
    ));
    Ok(decoder)
}

/// metaspace(replacement: String = "_", add_prefix_space: bool = true)
fn metaspace(mut cx: FunctionContext) -> JsResult<JsDecoder> {
    let replacement = cx.extract_opt::<char>(0)?.unwrap_or('▁');
    let add_prefix_space = cx.extract_opt::<bool>(1)?.unwrap_or(true);

    let mut decoder = JsDecoder::new::<_, JsDecoder, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    decoder.borrow_mut(&guard).decoder.make_owned(Box::new(
        tk::decoders::metaspace::Metaspace::new(replacement, add_prefix_space),
    ));
    Ok(decoder)
}

/// bpe_decoder(suffix: String = "</w>")
fn bpe_decoder(mut cx: FunctionContext) -> JsResult<JsDecoder> {
    let suffix = cx
        .extract_opt::<String>(0)?
        .unwrap_or_else(|| String::from("</w>"));

    let mut decoder = JsDecoder::new::<_, JsDecoder, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    decoder
        .borrow_mut(&guard)
        .decoder
        .make_owned(Box::new(tk::decoders::bpe::BPEDecoder::new(suffix)));
    Ok(decoder)
}

/// Register everything here
pub fn register(m: &mut ModuleContext, prefix: &str) -> NeonResult<()> {
    m.export_function(&format!("{}_ByteLevel", prefix), byte_level)?;
    m.export_function(&format!("{}_WordPiece", prefix), wordpiece)?;
    m.export_function(&format!("{}_Metaspace", prefix), metaspace)?;
    m.export_function(&format!("{}_BPEDecoder", prefix), bpe_decoder)?;
    Ok(())
}
