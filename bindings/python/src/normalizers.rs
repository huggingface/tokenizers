extern crate tokenizers as tk;

use super::utils::Container;
use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::*;

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

#[pyclass]
pub struct NFD {}
#[pymethods]
impl NFD {
    #[staticmethod]
    fn new() -> PyResult<Normalizer> {
        Ok(Normalizer {
            normalizer: Container::Owned(Box::new(tk::normalizers::unicode::NFD)),
        })
    }
}

#[pyclass]
pub struct NFKD {}
#[pymethods]
impl NFKD {
    #[staticmethod]
    fn new() -> PyResult<Normalizer> {
        Ok(Normalizer {
            normalizer: Container::Owned(Box::new(tk::normalizers::unicode::NFKD)),
        })
    }
}

#[pyclass]
pub struct NFC {}
#[pymethods]
impl NFC {
    #[staticmethod]
    fn new() -> PyResult<Normalizer> {
        Ok(Normalizer {
            normalizer: Container::Owned(Box::new(tk::normalizers::unicode::NFC)),
        })
    }
}

#[pyclass]
pub struct NFKC {}
#[pymethods]
impl NFKC {
    #[staticmethod]
    fn new() -> PyResult<Normalizer> {
        Ok(Normalizer {
            normalizer: Container::Owned(Box::new(tk::normalizers::unicode::NFKC)),
        })
    }
}

#[pyclass]
pub struct Sequence {}
#[pymethods]
impl Sequence {
    #[staticmethod]
    fn new(normalizers: &PyList) -> PyResult<Normalizer> {
        let normalizers = normalizers
            .iter()
            .map(|n| {
                let normalizer: &mut Normalizer = n.extract()?;
                if let Some(normalizer) = normalizer.normalizer.to_pointer() {
                    Ok(normalizer)
                } else {
                    Err(exceptions::Exception::py_err(
                        "At least one normalizer is already being used in another Tokenizer",
                    ))
                }
            })
            .collect::<PyResult<_>>()?;

        Ok(Normalizer {
            normalizer: Container::Owned(Box::new(tk::normalizers::utils::Sequence::new(
                normalizers,
            ))),
        })
    }
}

#[pyclass]
pub struct Lowercase {}
#[pymethods]
impl Lowercase {
    #[staticmethod]
    fn new() -> PyResult<Normalizer> {
        Ok(Normalizer {
            normalizer: Container::Owned(Box::new(tk::normalizers::utils::Lowercase)),
        })
    }
}
