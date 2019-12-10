extern crate tokenizers as tk;

use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::*;

use super::decoders::Decoder;
use super::encoding::Encoding;
use super::models::Model;
use super::pre_tokenizers::PreTokenizer;
use super::token::Token;
use super::trainers::Trainer;

#[pyclass(dict)]
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
                "The Model is already being used in another Tokenizer",
            ))
        }
    }

    fn with_model(&mut self, model: &mut Model) -> PyResult<()> {
        if let Some(model) = model.model.to_pointer() {
            self.tokenizer.with_model(model);
            Ok(())
        } else {
            Err(exceptions::Exception::py_err(
                "The Model is already being used in another Tokenizer",
            ))
        }
    }

    fn with_pre_tokenizer(&mut self, pretok: &mut PreTokenizer) -> PyResult<()> {
        if let Some(pretok) = pretok.pretok.to_pointer() {
            self.tokenizer.with_pre_tokenizer(pretok);
            Ok(())
        } else {
            Err(exceptions::Exception::py_err(
                "The PreTokenizer is already being used in another Tokenizer",
            ))
        }
    }

    fn with_decoder(&mut self, decoder: &mut Decoder) -> PyResult<()> {
        if let Some(decoder) = decoder.decoder.to_pointer() {
            self.tokenizer.with_decoder(decoder);
            Ok(())
        } else {
            Err(exceptions::Exception::py_err(
                "The Decoder is already being used in another Tokenizer",
            ))
        }
    }

    fn encode(&self, sentence: &str, pair: Option<&str>) -> Encoding {
        Encoding::new(self.tokenizer.encode(if pair.is_some() {
            tk::tokenizer::EncodeInput::Dual(sentence.to_owned(), pair.unwrap().to_owned())
        } else {
            tk::tokenizer::EncodeInput::Single(sentence.to_owned())
        }))
    }

    fn encode_batch(&self, sentences: &PyList) -> PyResult<Vec<Encoding>> {
        let inputs = sentences
            .into_iter()
            .map(|item| {
                if let Ok(s1) = item.extract::<String>() {
                    Ok(tk::tokenizer::EncodeInput::Single(s1))
                } else if let Ok((s1, s2)) = item.extract::<(String, String)>() {
                    Ok(tk::tokenizer::EncodeInput::Dual(s1, s2))
                } else {
                    Err(exceptions::Exception::py_err(
                        "Input must be a list[str] or list[(str, str)]",
                    ))
                }
            })
            .collect::<PyResult<Vec<_>>>()?;

        Ok(self
            .tokenizer
            .encode_batch(inputs)
            .into_iter()
            .map(|encoding| Encoding::new(encoding))
            .collect())
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

    fn train(&mut self, trainer: &Trainer, files: Vec<String>) -> PyResult<()> {
        trainer.trainer.execute(|trainer| {
            if let Err(e) = self.tokenizer.train(trainer, files) {
                Err(exceptions::Exception::py_err(format!("{}", e)))
            } else {
                Ok(())
            }
        })
    }
}
