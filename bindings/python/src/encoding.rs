extern crate tokenizers as tk;

use crate::error::PyError;
use pyo3::prelude::*;
use pyo3::types::*;
use pyo3::{PyObjectProtocol, PySequenceProtocol};
use tk::tokenizer::{Offsets, PaddingDirection};

#[pyclass(dict)]
#[repr(transparent)]
pub struct Encoding {
    pub encoding: tk::tokenizer::Encoding,
}

impl Encoding {
    pub fn new(encoding: tk::tokenizer::Encoding) -> Self {
        Encoding { encoding }
    }
}

#[pyproto]
impl PyObjectProtocol for Encoding {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "Encoding(num_tokens={}, attributes=[ids, type_ids, tokens, offsets, \
             attention_mask, special_tokens_mask, overflowing])",
            self.encoding.get_ids().len()
        ))
    }
}

#[pyproto]
impl PySequenceProtocol for Encoding {
    fn __len__(self) -> PyResult<usize> {
        Ok(self.encoding.get_ids().len())
    }
}

#[pymethods]
impl Encoding {
    #[staticmethod]
    #[args(growing_offsets = true)]
    fn merge(encodings: Vec<&Encoding>, growing_offsets: bool) -> Encoding {
        Encoding::new(tk::tokenizer::Encoding::merge(
            encodings
                .into_iter()
                .map(|e| e.encoding.clone())
                .collect::<Vec<_>>()
                .as_slice(),
            growing_offsets,
        ))
    }

    #[getter]
    fn get_ids(&self) -> Vec<u32> {
        self.encoding.get_ids().to_vec()
    }

    #[getter]
    fn get_tokens(&self) -> Vec<String> {
        self.encoding.get_tokens().to_vec()
    }

    #[getter]
    fn get_words(&self) -> Vec<u32> {
        self.encoding.get_words().to_vec()
    }

    #[getter]
    fn get_type_ids(&self) -> Vec<u32> {
        self.encoding.get_type_ids().to_vec()
    }

    #[getter]
    fn get_offsets(&self) -> Vec<(usize, usize)> {
        self.encoding.get_offsets().to_vec()
    }

    #[getter]
    fn get_special_tokens_mask(&self) -> Vec<u32> {
        self.encoding.get_special_tokens_mask().to_vec()
    }

    #[getter]
    fn get_attention_mask(&self) -> Vec<u32> {
        self.encoding.get_attention_mask().to_vec()
    }

    #[getter]
    fn get_overflowing(&self) -> Vec<Encoding> {
        self.encoding
            .get_overflowing()
            .clone()
            .into_iter()
            .map(Encoding::new)
            .collect()
    }

    fn char_to_word_offsets(&self, pos: usize) -> Option<Offsets> {
        self.encoding.char_to_word_offsets(pos)
    }

    fn char_to_token_offsets(&self, pos: usize) -> Option<Offsets> {
        self.encoding.char_to_token_offsets(pos)
    }

    fn token_to_word_offsets(&self, index: usize) -> Option<Offsets> {
        self.encoding.token_to_word_offsets(index)
    }

    fn char_to_token(&self, pos: usize) -> Option<usize> {
        self.encoding.char_to_token(pos)
    }

    #[args(kwargs = "**")]
    fn pad(&mut self, length: usize, kwargs: Option<&PyDict>) -> PyResult<()> {
        let mut pad_id = 0;
        let mut pad_type_id = 0;
        let mut pad_token = "[PAD]";
        let mut direction = PaddingDirection::Right;

        if let Some(kwargs) = kwargs {
            for (key, value) in kwargs {
                let key: &str = key.extract()?;
                match key {
                    "direction" => {
                        let value: &str = value.extract()?;
                        direction = match value {
                            "left" => Ok(PaddingDirection::Left),
                            "right" => Ok(PaddingDirection::Right),
                            other => Err(PyError(format!(
                                "Unknown `direction`: `{}`. Use \
                                 one of `left` or `right`",
                                other
                            ))
                            .into_pyerr()),
                        }?;
                    }
                    "pad_id" => pad_id = value.extract()?,
                    "pad_type_id" => pad_type_id = value.extract()?,
                    "pad_token" => pad_token = value.extract()?,
                    _ => println!("Ignored unknown kwarg option {}", key),
                }
            }
        }

        Ok(self
            .encoding
            .pad(length, pad_id, pad_type_id, pad_token, direction))
    }

    #[args(kwargs = "**")]
    fn truncate(&mut self, max_length: usize, kwargs: Option<&PyDict>) -> PyResult<()> {
        let mut stride = 0;

        if let Some(kwargs) = kwargs {
            for (key, value) in kwargs {
                let key: &str = key.extract()?;
                match key {
                    "stride" => stride = value.extract()?,
                    _ => println!("Ignored unknown kwarg option {}", key),
                }
            }
        }

        Ok(self.encoding.truncate(max_length, stride))
    }
}
