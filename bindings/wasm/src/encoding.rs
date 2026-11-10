use tokenizers::pipeline::Encoding as PipelineEncoding;
use wasm_bindgen::prelude::*;

/// Text encoded to token ids by a tokenizer.
#[wasm_bindgen]
pub struct Encoding(pub(crate) PipelineEncoding);

#[wasm_bindgen]
impl Encoding {
    /// The id of each token.
    #[wasm_bindgen(getter)]
    pub fn ids(&self) -> Vec<u32> {
        self.0.ids().iter().map(|token| token.id()).collect()
    }

    /// The type id of each token: 0 for the first sequence of a pair, 1 for the second.
    #[wasm_bindgen(getter)]
    pub fn type_ids(&self) -> Vec<u32> {
        widen(self.0.type_ids(), 0, self.0.len())
    }

    /// 1 for each token, 0 for each padding token.
    #[wasm_bindgen(getter)]
    pub fn attention_mask(&self) -> Vec<u32> {
        widen(self.0.attention_mask(), 1, self.0.len())
    }
}

// The core leaves out the all-zeros type ids and the all-ones mask.
fn widen(values: Option<&[u8]>, absent: u32, len: usize) -> Vec<u32> {
    match values {
        Some(values) => values.iter().map(|&value| value as u32).collect(),
        None => vec![absent; len],
    }
}
