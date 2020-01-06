extern crate tokenizers as tk;

use crate::error::PyError;
use crate::normalized_string::NormalizedString;
use pyo3::prelude::*;
use pyo3::types::*;
use tk::tokenizer::PaddingDirection;

#[pyclass(dict)]
#[repr(transparent)]
pub struct Encoding {
    encoding: tk::tokenizer::Encoding,
}

impl Encoding {
    pub fn new(encoding: tk::tokenizer::Encoding) -> Self {
        Encoding { encoding }
    }
}

#[pymethods]
impl Encoding {
    #[getter]
    fn get_normalized(&self) -> NormalizedString {
        NormalizedString::new(self.encoding.get_normalized().clone())
    }

    #[args(kwargs = "**")]
    fn get_range(
        &self,
        range: (usize, usize),
        kwargs: Option<&PyDict>,
    ) -> PyResult<Option<String>> {
        let mut original = false;
        if let Some(kwargs) = kwargs {
            if let Some(koriginal) = kwargs.get_item("original") {
                original = koriginal.extract()?;
            }
        }

        if original {
            Ok(self
                .encoding
                .get_normalized()
                .get_range_original(range.0..range.1)
                .map(|s| s.to_owned()))
        } else {
            Ok(self
                .encoding
                .get_normalized()
                .get_range(range.0..range.1)
                .map(|s| s.to_owned()))
        }
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
    fn get_overflowing(&self) -> Option<Encoding> {
        self.encoding.get_overflowing().cloned().map(Encoding::new)
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
            .pad(length, pad_id, pad_type_id, pad_token, &direction))
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
