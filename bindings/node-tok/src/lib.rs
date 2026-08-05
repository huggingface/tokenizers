//! A Node binding whose entire surface is the `.tok` read path.
//!
//! There is deliberately no `fromFile`, no `fromString`, no config accessors: the moment one
//! exists, `serde_json` is reachable and LTO has to keep the whole JSON stack. Converting a
//! `tokenizer.json` is `tk-convert`'s job, offline.

use napi::bindgen_prelude::*;
use napi_derive::napi;

use tk_encode::pipeline::PipelineTokenizer as Pipeline;

#[napi]
pub struct TokTokenizer {
    inner: Pipeline,
    // Keeps the mapped bytes alive for as long as the tokenizer that was built from them.
    _file: tk_serialization::TokFile,
}

#[napi]
impl TokTokenizer {
    /// Load a `.tok` produced by `tk-convert`.
    #[napi(factory)]
    pub fn from_file(path: String) -> Result<Self> {
        let file = tk_serialization::TokFile::open(&path)
            .map_err(|e| Error::from_reason(format!("{path}: {e}")))?;
        let inner = Pipeline::from_tok(file.bytes())
            .map_err(|e| Error::from_reason(format!("{path}: {e}")))?;
        Ok(Self { inner, _file: file })
    }

    /// Encode `text`, returning the ids. A `Uint32Array` rather than a `Vec<u32>`: the latter
    /// marshals as a boxed JS array, one napi value per token, which costs more than the encode.
    #[napi]
    pub fn encode(&self, text: String, add_special_tokens: bool) -> Result<Uint32Array> {
        let encoded = self
            .inner
            .encode(text.as_str(), add_special_tokens)
            .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(encoded.iter().map(|t| t.id).collect::<Vec<u32>>().into())
    }

    /// Encode UTF-8 bytes straight into a caller-owned buffer, returning how many ids were
    /// written. Drops the JS string copy and the fresh ArrayBuffer.
    #[napi]
    pub fn encode_bytes_into(
        &self,
        text: Buffer,
        mut out: Uint32Array,
        add_special_tokens: bool,
    ) -> Result<u32> {
        let text = std::str::from_utf8(&text)
            .map_err(|e| Error::from_reason(format!("input is not valid UTF-8: {e}")))?;
        let encoded = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| Error::from_reason(e.to_string()))?;
        if encoded.len() > out.len() {
            return Err(Error::from_reason(format!(
                "output buffer holds {} ids, needs {}",
                out.len(),
                encoded.len()
            )));
        }
        // SAFETY: `out` is a JS-owned `Uint32Array` handed to this call; napi only marks the
        // mutable view unsafe because JS could alias it, and nothing here re-enters JS.
        let slots = unsafe { out.as_mut() };
        for (slot, token) in slots.iter_mut().zip(&encoded) {
            *slot = token.id;
        }
        Ok(encoded.len() as u32)
    }
}
