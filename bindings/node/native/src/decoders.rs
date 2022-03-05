extern crate tokenizers as tk;

use crate::extraction::*;
use neon::prelude::*;
use std::sync::Arc;

use tk::decoders::DecoderWrapper;

/// Decoder
#[derive(Clone, Serialize, Deserialize)]
pub struct Decoder {
    #[serde(flatten)]
    pub decoder: Option<Arc<DecoderWrapper>>,
}

impl tk::Decoder for Decoder {
    fn decode_chain(&self, tokens: Vec<String>) -> tk::Result<Vec<String>> {
        self.decoder
            .as_ref()
            .ok_or("Uninitialized Decoder")?
            .decode_chain(tokens)
    }
}

declare_types! {
    pub class JsDecoder for Decoder {
        init(_) {
             // This should not be called from JS
             Ok(Decoder { decoder: None })
        }

        method decode(mut cx) {
            use tk::Decoder;

            let tokens = cx.extract_vec::<String>(0)?;

            let this = cx.this();
            let guard = cx.lock();
            let output = this.borrow(&guard)
                .decoder.as_ref().unwrap()
                .decode(tokens)
                .map_err(|e| Error(format!("{}", e)))?;

            Ok(cx.string(output).upcast())

        }
    }
}

/// byte_level()
fn byte_level(mut cx: FunctionContext) -> JsResult<JsDecoder> {
    let mut decoder = JsDecoder::new::<_, JsDecoder, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    decoder.borrow_mut(&guard).decoder = Some(Arc::new(
        tk::decoders::byte_level::ByteLevel::default().into(),
    ));
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
    decoder.borrow_mut(&guard).decoder = Some(Arc::new(
        tk::decoders::wordpiece::WordPiece::new(prefix, cleanup).into(),
    ));
    Ok(decoder)
}

/// metaspace(replacement: String = "_", add_prefix_space: bool = true)
fn metaspace(mut cx: FunctionContext) -> JsResult<JsDecoder> {
    let replacement = cx.extract_opt::<char>(0)?.unwrap_or('▁');
    let add_prefix_space = cx.extract_opt::<bool>(1)?.unwrap_or(true);

    let mut decoder = JsDecoder::new::<_, JsDecoder, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    decoder.borrow_mut(&guard).decoder = Some(Arc::new(
        tk::decoders::metaspace::Metaspace::new(replacement, add_prefix_space).into(),
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
    decoder.borrow_mut(&guard).decoder =
        Some(Arc::new(tk::decoders::bpe::BPEDecoder::new(suffix).into()));
    Ok(decoder)
}

/// ctc_decoder(pad_token: String = "<pad>", word_delimiter_token: String = "|", cleanup = true)
fn ctc_decoder(mut cx: FunctionContext) -> JsResult<JsDecoder> {
    let pad_token = cx
        .extract_opt::<String>(0)?
        .unwrap_or_else(|| String::from("<pad>"));
    let word_delimiter_token = cx
        .extract_opt::<String>(1)?
        .unwrap_or_else(|| String::from("|"));
    let cleanup = cx.extract_opt::<bool>(2)?.unwrap_or(true);

    let mut decoder = JsDecoder::new::<_, JsDecoder, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    decoder.borrow_mut(&guard).decoder = Some(Arc::new(
        tk::decoders::ctc::CTC::new(pad_token, word_delimiter_token, cleanup).into(),
    ));
    Ok(decoder)
}

/// sequence()
fn sequence(mut cx: FunctionContext) -> JsResult<JsDecoder> {
    let decoders = cx.argument::<JsArray>(0)?.to_vec(&mut cx)?;
    let mut sequence = Vec::with_capacity(decoders.len());

    decoders.into_iter().try_for_each(|decoder| {
        match decoder.downcast::<JsDecoder>().or_throw(&mut cx) {
            Ok(decoder) => {
                let guard = cx.lock();
                if let Some(decoder_arc) = &decoder.borrow(&guard).decoder {
                    let decoder: DecoderWrapper = (**decoder_arc).clone();
                    sequence.push(decoder);
                }
                Ok(())
            }
            Err(e) => Err(e),
        }
    })?;

    let mut pretok = JsDecoder::new::<_, JsDecoder, _>(&mut cx, vec![])?;
    let guard = cx.lock();
    pretok.borrow_mut(&guard).decoder = Some(Arc::new(tk::DecoderWrapper::Sequence(
        tk::decoders::sequence::Sequence::new(sequence),
    )));
    Ok(pretok)
}

/// Register everything here
pub fn register(m: &mut ModuleContext, prefix: &str) -> NeonResult<()> {
    m.export_function(&format!("{}_ByteLevel", prefix), byte_level)?;
    m.export_function(&format!("{}_WordPiece", prefix), wordpiece)?;
    m.export_function(&format!("{}_Metaspace", prefix), metaspace)?;
    m.export_function(&format!("{}_BPEDecoder", prefix), bpe_decoder)?;
    m.export_function(&format!("{}_CTC", prefix), ctc_decoder)?;
    m.export_function(&format!("{}_Sequence", prefix), sequence)?;
    Ok(())
}
