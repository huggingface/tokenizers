use tk_convert::canonicalize_str;
use tk_encode::pipeline::PipelineTokenizer;
use tk_serialize::from_json;

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Tokenizer(PipelineTokenizer);

#[wasm_bindgen]
impl Tokenizer {
    pub fn from_json(json: &str) -> Result<Self, JsError> {
        let canonical = canonicalize_str(json)?;
        let tok = from_json(&canonical).map_err(js_err)?;
        Ok(Self(tok))
    }
}

fn js_err(err: Box<dyn std::error::Error + Send + Sync>) -> JsError {
    JsError::new(&err.to_string())
}
