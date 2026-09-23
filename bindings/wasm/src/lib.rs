use tk_convert::canonicalize_str;
use tk_encode::pipeline::PipelineTokenizer;
use tk_serialize::from_json;

use wasm_bindgen::prelude::*;

mod options;

use options::convert_encode_options;

pub use options::{EncodeOptions, PaddingOptions, TruncationOptions};

#[wasm_bindgen]
/// A tokenizer loaded from a `tokenizer.json`.
///
/// It lives in WebAssembly memory. 
/// Call {@link Tokenizer.free | free()} to free the object when you are done using it, or
/// declare it with `using` to free it at the end of the scope.
///
/// @example
/// ```ts
/// using tokenizer = Tokenizer.from_json(json);
/// const ids = tokenizer.encode("Hello world");
/// ```
pub struct Tokenizer(PipelineTokenizer);

#[wasm_bindgen]
impl Tokenizer {
    /// Loads a tokenizer from the contents of a `tokenizer.json`.
    ///
    /// @param json - The contents of a `tokenizer.json` file.
    /// @returns The tokenizer.
    /// @throws If `json` is not valid JSON or does not describe a tokenizer.
    pub fn from_json(json: &str) -> Result<Self, JsError> {
        let canonical = canonicalize_str(json)?;
        let tok = from_json(&canonical).map_err(js_err)?;
        Ok(Self(tok))
    }

    /// Encodes the given text to token ids.
    ///
    /// @param text - The text to encode into tokens.
    /// @param options - Settings for this call: whether to add special tokens, padding or
    /// truncation. Defaults to the tokenizer's config. See {@link EncodeOptions}.
    /// @returns The token ids.
    /// @throws If an option has an unknown value.
    pub fn encode(&self, text: &str, options: Option<EncodeOptions>) -> Result<Vec<u32>, JsError> {
        let encodings = self
            .0
            .encode(text, &convert_encode_options(options)?)
            .wait()
            .map_err(js_err)?;
        let ids = encodings
            .first()
            .map(|e| e.ids().iter().map(|t| t.id()).collect())
            .unwrap_or_default();
        Ok(ids)
    }

    /// Decodes token ids back into text.
    ///
    /// @param ids - The ids to convert. Can be the result of {@link Tokenizer.encode}.
    /// @param skip_special_tokens - Whether special tokens are ignored. `false` when not specified.
    /// @returns The decoded text.
    pub fn decode(
        &self,
        ids: &[u32],
        skip_special_tokens: Option<bool>,
    ) -> Result<String, JsError> {
        self.0
            .decode(ids, skip_special_tokens.unwrap_or(true))
            .map_err(js_err)
    }

    /// Converts token ids to their tokens. This does not run the decoder, so each token keeps its
    /// spelling in the vocabulary: a byte-level BPE token reads `Ġworld`, not ` world`.
    ///
    /// @param ids - The ids to convert. Can be the result of {@link Tokenizer.encode}.
    /// @param skip_special_tokens - Whether special tokens are ignored. `false` when not specified.
    /// @returns The token of each id found in the vocabulary. Ids not in the vocabulary are dropped.
    pub fn decode_tokens(&self, ids: &[u32], skip_special_tokens: Option<bool>) -> Vec<String> {
        self.0
            .decode_tokens(ids, skip_special_tokens.unwrap_or(false))
    }
}

fn js_err(err: Box<dyn std::error::Error + Send + Sync>) -> JsError {
    JsError::new(&err.to_string())
}
