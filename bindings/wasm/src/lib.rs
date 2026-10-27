use tokenizers::canonicalize_str;
use tokenizers::from_json;
use tokenizers::pipeline::PipelineTokenizer;
use tsify::Ts;

use wasm_bindgen::prelude::*;

mod encoding;
mod hub;
mod input;
mod options;

use options::convert_encode_options;

pub use encoding::Encoding;
pub use hub::PretrainedOptions;
pub use input::TextInput;
pub use options::EncodeOptions;

#[wasm_bindgen]
/// A tokenizer loaded from a `tokenizer.json`.
///
/// It lives in WebAssembly memory, which the garbage collector frees once the tokenizer is unreachable.
/// To free it right away, call {@link Tokenizer.free | free()} or declare it with `using`.
///
/// @example
/// ```ts
/// using tokenizer = Tokenizer.from_json(json);
/// const { ids } = tokenizer.encode("Hello world");
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

    /// Downloads `tokenizer.json` from a repo on the Hugging Face Hub and loads it.
    ///
    /// @param repoName - The id of a model repo, such as `"openai-community/gpt2"`.
    /// @param options - See {@link PretrainedOptions}.
    /// @returns The tokenizer.
    /// @throws If the download fails, or the file does not describe a tokenizer.
    ///
    /// @example
    /// ```ts
    /// using tokenizer = await Tokenizer.from_pretrained("openai-community/gpt2");
    /// using pinned = await Tokenizer.from_pretrained("openai-community/gpt2", { revision: "607a30d" });
    /// using private = await Tokenizer.from_pretrained("user/private-model", { token: "hf_xxxxxxxxxxxxxxxxxxxxxxxx" });
    /// ```
    pub async fn from_pretrained(
        repo_name: String,
        options: Option<Ts<PretrainedOptions>>,
    ) -> Result<Tokenizer, JsValue> {
        let json = hub::fetch_tokenizer_json(&repo_name, options).await?;
        Ok(Self::from_json(&json)?)
    }

    /// Encodes the given text, or pair of texts, to token ids.
    ///
    /// @param input - A text, or a pair of texts given as a two-item array. The post-processor
    /// joins a pair, for example as `[CLS] A [SEP] B [SEP]`.
    /// @param options - Settings for this call: whether to add special tokens, padding or
    /// truncation. Defaults to the tokenizer's config. See {@link EncodeOptions}.
    /// @returns The {@link Encoding}.
    /// @throws If the input is not a text or a pair, or the options do not match {@link EncodeOptions}.
    pub fn encode(
        &self,
        input: Ts<TextInput>,
        options: Option<Ts<EncodeOptions>>,
    ) -> Result<Encoding, JsError> {
        let input = input.to_rust()?;
        let options = convert_encode_options(options)?;
        let encoding = self
            .0
            .encode(input, &options)
            .wait()
            .map_err(js_err)?
            .pop()
            .expect("encoding a single input gives exactly one encoding");
        Ok(Encoding(encoding))
    }

    /// Decodes token ids back into text.
    ///
    /// @param ids - The ids to convert, such as {@link Encoding.ids}.
    /// @param skip_special_tokens - Whether special tokens are ignored. `true` when not specified.
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
    /// @param ids - The ids to convert, such as {@link Encoding.ids}.
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
