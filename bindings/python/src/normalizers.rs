extern crate tokenizers as tk;

use super::utils::Container;
use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::*;

#[pyclass(dict)]
pub struct Normalizer {
    pub normalizer: Container<dyn tk::tokenizer::Normalizer>,
}

#[pyclass(extends=Normalizer)]
pub struct BertNormalizer {}
#[pymethods]
impl BertNormalizer {
    #[new]
    #[args(kwargs = "**")]
    fn new(kwargs: Option<&PyDict>) -> PyResult<(Self, Normalizer)> {
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

        Ok((
            BertNormalizer {},
            Normalizer {
                normalizer: Container::Owned(Box::new(tk::normalizers::bert::BertNormalizer::new(
                    clean_text,
                    handle_chinese_chars,
                    strip_accents,
                    lowercase,
                ))),
            },
        ))
    }
}

#[pyclass(extends=Normalizer)]
pub struct NFD {}
#[pymethods]
impl NFD {
    #[new]
    fn new() -> PyResult<(Self, Normalizer)> {
        Ok((
            NFD {},
            Normalizer {
                normalizer: Container::Owned(Box::new(tk::normalizers::unicode::NFD)),
            },
        ))
    }
}

#[pyclass(extends=Normalizer)]
pub struct NFKD {}
#[pymethods]
impl NFKD {
    #[new]
    fn new() -> PyResult<(Self, Normalizer)> {
        Ok((
            NFKD {},
            Normalizer {
                normalizer: Container::Owned(Box::new(tk::normalizers::unicode::NFKD)),
            },
        ))
    }
}

#[pyclass(extends=Normalizer)]
pub struct NFC {}
#[pymethods]
impl NFC {
    #[new]
    fn new() -> PyResult<(Self, Normalizer)> {
        Ok((
            NFC {},
            Normalizer {
                normalizer: Container::Owned(Box::new(tk::normalizers::unicode::NFC)),
            },
        ))
    }
}

#[pyclass(extends=Normalizer)]
pub struct NFKC {}
#[pymethods]
impl NFKC {
    #[new]
    fn new() -> PyResult<(Self, Normalizer)> {
        Ok((
            NFKC {},
            Normalizer {
                normalizer: Container::Owned(Box::new(tk::normalizers::unicode::NFKC)),
            },
        ))
    }
}

#[pyclass(extends=Normalizer)]
pub struct Sequence {}
#[pymethods]
impl Sequence {
    #[new]
    fn new(normalizers: &PyList) -> PyResult<(Self, Normalizer)> {
        let normalizers = normalizers
            .iter()
            .map(|n| {
                let mut normalizer: PyRefMut<Normalizer> = n.extract()?;
                if let Some(normalizer) = normalizer.normalizer.to_pointer() {
                    Ok(normalizer)
                } else {
                    Err(exceptions::Exception::py_err(
                        "At least one normalizer is already being used in another Tokenizer",
                    ))
                }
            })
            .collect::<PyResult<_>>()?;

        Ok((
            Sequence {},
            Normalizer {
                normalizer: Container::Owned(Box::new(tk::normalizers::utils::Sequence::new(
                    normalizers,
                ))),
            },
        ))
    }
}

#[pyclass(extends=Normalizer)]
pub struct Lowercase {}
#[pymethods]
impl Lowercase {
    #[new]
    fn new() -> PyResult<(Self, Normalizer)> {
        Ok((
            Lowercase {},
            Normalizer {
                normalizer: Container::Owned(Box::new(tk::normalizers::utils::Lowercase)),
            },
        ))
    }
}

#[pyclass(extends=Normalizer)]
pub struct Strip {}
#[pymethods]
impl Strip {
    #[new]
    #[args(kwargs = "**")]
    fn new(kwargs: Option<&PyDict>) -> PyResult<(Self, Normalizer)> {
        let mut left = true;
        let mut right = true;

        if let Some(kwargs) = kwargs {
            if let Some(l) = kwargs.get_item("left") {
                left = l.extract()?;
            }
            if let Some(r) = kwargs.get_item("right") {
                right = r.extract()?;
            }
        }

        Ok((
            Strip {},
            Normalizer {
                normalizer: Container::Owned(Box::new(tk::normalizers::strip::Strip::new(
                    left, right,
                ))),
            },
        ))
    }
}
