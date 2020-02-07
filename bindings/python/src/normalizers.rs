extern crate tokenizers as tk;

use super::utils::Container;
use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::*;

#[pyclass(dict)]
pub struct Normalizer {
    pub normalizer: Container<dyn tk::tokenizer::Normalizer + Sync>,
}

#[pyclass(extends=Normalizer)]
pub struct BertNormalizer {}
#[pymethods]
impl BertNormalizer {
    #[new]
    #[args(kwargs = "**")]
    fn new(obj: &PyRawObject, kwargs: Option<&PyDict>) -> PyResult<()> {
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

        Ok(obj.init(Normalizer {
            normalizer: Container::Owned(Box::new(tk::normalizers::bert::BertNormalizer::new(
                clean_text,
                handle_chinese_chars,
                strip_accents,
                lowercase,
            ))),
        }))
    }
}

#[pyclass(extends=Normalizer)]
pub struct NFD {}
#[pymethods]
impl NFD {
    #[new]
    fn new(obj: &PyRawObject) -> PyResult<()> {
        Ok(obj.init(Normalizer {
            normalizer: Container::Owned(Box::new(tk::normalizers::unicode::NFD)),
        }))
    }
}

#[pyclass(extends=Normalizer)]
pub struct NFKD {}
#[pymethods]
impl NFKD {
    #[new]
    fn new(obj: &PyRawObject) -> PyResult<()> {
        Ok(obj.init(Normalizer {
            normalizer: Container::Owned(Box::new(tk::normalizers::unicode::NFKD)),
        }))
    }
}

#[pyclass(extends=Normalizer)]
pub struct NFC {}
#[pymethods]
impl NFC {
    #[new]
    fn new(obj: &PyRawObject) -> PyResult<()> {
        Ok(obj.init(Normalizer {
            normalizer: Container::Owned(Box::new(tk::normalizers::unicode::NFC)),
        }))
    }
}

#[pyclass(extends=Normalizer)]
pub struct NFKC {}
#[pymethods]
impl NFKC {
    #[new]
    fn new(obj: &PyRawObject) -> PyResult<()> {
        Ok(obj.init(Normalizer {
            normalizer: Container::Owned(Box::new(tk::normalizers::unicode::NFKC)),
        }))
    }
}

#[pyclass(extends=Normalizer)]
pub struct Sequence {}
#[pymethods]
impl Sequence {
    #[new]
    fn new(obj: &PyRawObject, normalizers: &PyList) -> PyResult<()> {
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

        Ok(obj.init(Normalizer {
            normalizer: Container::Owned(Box::new(tk::normalizers::utils::Sequence::new(
                normalizers,
            ))),
        }))
    }
}

#[pyclass(extends=Normalizer)]
pub struct Lowercase {}
#[pymethods]
impl Lowercase {
    #[new]
    fn new(obj: &PyRawObject) -> PyResult<()> {
        Ok(obj.init(Normalizer {
            normalizer: Container::Owned(Box::new(tk::normalizers::utils::Lowercase)),
        }))
    }
}
