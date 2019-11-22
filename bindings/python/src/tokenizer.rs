extern crate tokenizers as tk;

use pyo3::exceptions;
use pyo3::prelude::*;

use super::models::Model;
use super::token::Token;

#[pyclass]
pub struct Tokenizer {
    tokenizer: tk::tokenizer::Tokenizer,
}

#[pymethods]
impl Tokenizer {
    #[new]
    fn new(obj: &PyRawObject, model: &mut Model) -> PyResult<()> {
        if let Some(model) = model.model.to_pointer() {
            let tokenizer = tk::tokenizer::Tokenizer::new(model);
            obj.init({ Tokenizer { tokenizer } });
            Ok(())
        } else {
            Err(exceptions::Exception::py_err(
                "The model is already being used in another Tokenizer",
            ))
        }
    }

    fn with_model(&mut self, model: &mut Model) -> PyResult<()> {
        if let Some(model) = model.model.to_pointer() {
            self.tokenizer.with_model(model);
            Ok(())
        } else {
            Err(exceptions::Exception::py_err(
                "The model is already being used in another Tokenizer",
            ))
        }
    }

    fn encode(&self, sentence: &str) -> Vec<Token> {
        self.tokenizer
            .encode(sentence)
            .into_iter()
            .map(|token| Token::new(token))
            .collect()
    }

    fn encode_batch(&self, sentences: Vec<&str>) -> Vec<Vec<Token>> {
        self.tokenizer
            .encode_batch(sentences)
            .into_iter()
            .map(|sentence| {
                sentence
                    .into_iter()
                    .map(|token| Token::new(token))
                    .collect()
            })
            .collect()
    }

    fn decode(&self, ids: Vec<u32>) -> String {
        self.tokenizer.decode(ids)
    }

    fn decode_batch(&self, sentences: Vec<Vec<u32>>) -> Vec<String> {
        self.tokenizer.decode_batch(sentences)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.tokenizer.token_to_id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.tokenizer.id_to_token(id)
    }
}

