use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::*;
use pyo3::{PyObjectProtocol, PySequenceProtocol};
use tk::tokenizer::{Offsets, PaddingDirection};
use tokenizers as tk;

use crate::error::PyError;

#[pyclass(dict, module = "tokenizers", name=Encoding)]
#[repr(transparent)]
pub struct PyEncoding {
    pub encoding: tk::tokenizer::Encoding,
}

impl From<tk::tokenizer::Encoding> for PyEncoding {
    fn from(v: tk::tokenizer::Encoding) -> Self {
        Self { encoding: v }
    }
}

#[pyproto]
impl PyObjectProtocol for PyEncoding {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "Encoding(num_tokens={}, attributes=[ids, type_ids, tokens, offsets, \
             attention_mask, special_tokens_mask, overflowing])",
            self.encoding.get_ids().len()
        ))
    }
}

#[pyproto]
impl PySequenceProtocol for PyEncoding {
    fn __len__(self) -> PyResult<usize> {
        Ok(self.encoding.len())
    }
}

#[pymethods]
impl PyEncoding {
    #[new]
    fn new() -> PyResult<Self> {
        Ok(Self {
            encoding: tk::tokenizer::Encoding::default(),
        })
    }

    fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        let data = serde_json::to_string(&self.encoding).map_err(|e| {
            exceptions::PyException::new_err(format!(
                "Error while attempting to pickle Encoding: {}",
                e.to_string()
            ))
        })?;
        Ok(PyBytes::new(py, data.as_bytes()).to_object(py))
    }

    fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        match state.extract::<&PyBytes>(py) {
            Ok(s) => {
                self.encoding = serde_json::from_slice(s.as_bytes()).map_err(|e| {
                    exceptions::PyException::new_err(format!(
                        "Error while attempting to unpickle Encoding: {}",
                        e.to_string()
                    ))
                })?;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    #[staticmethod]
    #[args(growing_offsets = true)]
    fn merge(encodings: Vec<PyRef<PyEncoding>>, growing_offsets: bool) -> PyEncoding {
        tk::tokenizer::Encoding::merge(
            encodings.into_iter().map(|e| e.encoding.clone()),
            growing_offsets,
        )
        .into()
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
    fn get_words(&self) -> Vec<Option<u32>> {
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
    fn get_overflowing(&self) -> Vec<PyEncoding> {
        self.encoding
            .get_overflowing()
            .clone()
            .into_iter()
            .map(|e| e.into())
            .collect()
    }

    fn word_to_tokens(&self, word_index: u32) -> Option<(usize, usize)> {
        self.encoding.word_to_tokens(word_index)
    }

    fn word_to_chars(&self, word_index: u32) -> Option<Offsets> {
        self.encoding.word_to_chars(word_index)
    }

    fn token_to_chars(&self, token_index: usize) -> Option<Offsets> {
        self.encoding.token_to_chars(token_index)
    }

    fn token_to_word(&self, token_index: usize) -> Option<u32> {
        self.encoding.token_to_word(token_index)
    }

    fn char_to_token(&self, char_pos: usize) -> Option<usize> {
        self.encoding.char_to_token(char_pos)
    }

    fn char_to_word(&self, char_pos: usize) -> Option<u32> {
        self.encoding.char_to_word(char_pos)
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
                            .into_pyerr::<exceptions::PyValueError>()),
                        }?;
                    }
                    "pad_id" => pad_id = value.extract()?,
                    "pad_type_id" => pad_type_id = value.extract()?,
                    "pad_token" => pad_token = value.extract()?,
                    _ => println!("Ignored unknown kwarg option {}", key),
                }
            }
        }
        self.encoding
            .pad(length, pad_id, pad_type_id, pad_token, direction);
        Ok(())
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
        self.encoding.truncate(max_length, stride);
        Ok(())
    }
}
