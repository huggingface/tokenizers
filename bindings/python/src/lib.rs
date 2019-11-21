extern crate tokenizers as tk;
use tk::models::bpe::Error as BpeError;

use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

#[pyclass]
#[repr(transparent)]
struct Token {
    tok: tk::tokenizer::Token,
}
impl Token {
    pub fn new(tok: tk::tokenizer::Token) -> Self {
        Token { tok }
    }
}

#[pymethods]
impl Token {
    #[getter]
    fn get_id(&self) -> PyResult<u32> {
        Ok(self.tok.id)
    }

    #[getter]
    fn get_value(&self) -> PyResult<&str> {
        Ok(&self.tok.value)
    }

    #[getter]
    fn get_offsets(&self) -> PyResult<(usize, usize)> {
        Ok(self.tok.offsets)
    }

    fn as_tuple(&self) -> PyResult<(u32, &str, (usize, usize))> {
        Ok((self.tok.id, &self.tok.value, self.tok.offsets))
    }
}

fn get_pre_tokenizer(name: &str) -> Option<Box<dyn tk::tokenizer::PreTokenizer + Sync>> {
    match name {
        "ByteLevel" => Some(Box::new(tk::pre_tokenizers::byte_level::ByteLevel)),
        "Whitespace" => Some(Box::new(tk::pre_tokenizers::whitespace::Whitespace)),
        _ => None,
    }
}

fn get_normalizer(_name: &str) -> Option<Box<dyn tk::tokenizer::Normalizer + Sync>> {
    None
}

fn get_post_processor(_name: &str) -> Option<Box<dyn tk::tokenizer::PostProcessor + Sync>> {
    None
}

fn get_decoder(name: &str) -> Option<Box<dyn tk::tokenizer::Decoder + Sync>> {
    match name {
        "ByteLevel" => Some(Box::new(tk::decoders::byte_level::ByteLevel)),
        _ => None,
    }
}

#[pyclass]
struct Tokenizer {
    tokenizer: tk::tokenizer::Tokenizer,
}
#[pymethods]
impl Tokenizer {
    #[staticmethod]
    #[args(kwargs = "**")]
    fn bpe_from_files(vocab: &str, merges: &str, kwargs: Option<&PyDict>) -> PyResult<Self> {
        let model = match tk::models::bpe::BPE::from_files(vocab, merges) {
            Ok(bpe) => Ok(Box::new(bpe)),
            Err(e) => match e {
                BpeError::BadVocabulary => {
                    Err(exceptions::Exception::py_err("Bad vocab.json format"))
                }
                BpeError::Io(io) => Err(PyErr::from(io)),
                BpeError::JsonError(_) => Err(exceptions::Exception::py_err(
                    "Error while parsing vocab json file",
                )),
                BpeError::MergeTokenOutOfVocabulary(token) => Err(exceptions::Exception::py_err(
                    format!("Merge token out of vocabulary: {}", token),
                )),
            },
        }?;

        let mut tokenizer = tk::tokenizer::Tokenizer::new(model);

        if let Some(kwargs) = kwargs {
            for (option, value) in kwargs {
                match option.to_string().as_ref() {
                    "pre_tokenizer" => {
                        let value = value.to_string();
                        if let Some(pre_tokenizer) = get_pre_tokenizer(&value) {
                            tokenizer.with_pre_tokenizer(pre_tokenizer);
                        } else {
                            return Err(exceptions::Exception::py_err(format!(
                                "PreTokenizer `{}` not found",
                                value
                            )));
                        }
                    }
                    "normalizers" => {
                        let mut normalizers = vec![];
                        let values = value.cast_as::<PyList>()?;
                        for value in values {
                            let value = value.to_string();
                            if let Some(normalizer) = get_normalizer(&value) {
                                normalizers.push(normalizer);
                            } else {
                                return Err(exceptions::Exception::py_err(format!(
                                    "Normalizer `{}` not found",
                                    value
                                )));
                            }
                        }
                        tokenizer.with_normalizers(normalizers);
                    }
                    "post_processors" => {
                        let mut processors = vec![];
                        let values = value.cast_as::<PyList>()?;
                        for value in values {
                            let value = value.to_string();
                            if let Some(processor) = get_post_processor(&value) {
                                processors.push(processor);
                            } else {
                                return Err(exceptions::Exception::py_err(format!(
                                    "PostProcessor `{}` not found",
                                    value
                                )));
                            }
                        }
                        tokenizer.with_post_processors(processors);
                    }
                    "decoder" => {
                        let value = value.to_string();
                        if let Some(decoder) = get_decoder(&value) {
                            tokenizer.with_decoder(decoder);
                        } else {
                            return Err(exceptions::Exception::py_err(format!(
                                "Decoder `{}` not found",
                                value
                            )));
                        }
                    }
                    _ => println!("Ignored unknown kwarg `{}`", option),
                }
            }
        }

        Ok(Tokenizer { tokenizer })
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

#[pymodule]
fn tokenizers(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Tokenizer>()?;
    Ok(())
}
