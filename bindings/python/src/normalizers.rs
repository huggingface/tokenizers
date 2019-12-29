extern crate tokenizers as tk;

use super::error::{PyError, ToPyResult};
use super::utils::Container;
use pyo3::prelude::*;
use pyo3::types::*;
use tk::tokenizer::Result;

#[pyclass(dict)]
pub struct Normalizer {
    pub normalizer: Container<dyn tk::tokenizer::Normalizer + Sync>,
}

#[pyclass]
pub struct BertNormalizer {}
#[pymethods]
impl BertNormalizer {
    #[staticmethod]
    #[args(kwargs = "**")]
    fn new(kwargs: Option<&PyDict>) -> PyResult<Normalizer> {
        let mut clean_text = true;
        let mut handle_chinese_chars = true;
        let mut strip_accents = true;
        let mut lowercase = true;

        if let Some(kwargs) = kwargs {
            for (key, value) in kwargs {
                let key: &str = key.extract()?;
                match key {
                    "clean_text" => clean_text = value.extract()?,
                    "handle_chinese_chars" => handle_chinese_chars = value.extract()?,
                    "strip_accents" => strip_accents = value.extract()?,
                    "lowercase" => lowercase = value.extract()?,
                    _ => println!("Ignored unknown kwargs option {}", key),
                }
            }
        }

        Ok(Normalizer {
            normalizer: Container::Owned(Box::new(tk::normalizers::bert::BertNormalizer::new(
                clean_text,
                handle_chinese_chars,
                strip_accents,
                lowercase,
            ))),
        })
    }
}
