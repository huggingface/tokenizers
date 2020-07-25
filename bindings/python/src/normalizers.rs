use std::sync::Arc;

use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::*;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use tk::normalizers::bert::BertNormalizer;
use tk::normalizers::strip::Strip;
use tk::normalizers::unicode::{NFC, NFD, NFKC, NFKD};
use tk::normalizers::utils::{Lowercase, Sequence};
use tk::{NormalizedString, Normalizer};
use tokenizers as tk;

#[pyclass(dict, module = "tokenizers.normalizers", name=Normalizer)]
#[derive(Clone)]
pub struct PyNormalizer {
    pub normalizer: Arc<dyn Normalizer>,
}

impl PyNormalizer {
    pub fn new(normalizer: Arc<dyn Normalizer>) -> Self {
        PyNormalizer { normalizer }
    }
}

#[typetag::serde]
impl Normalizer for PyNormalizer {
    fn normalize(&self, normalized: &mut NormalizedString) -> tk::Result<()> {
        self.normalizer.normalize(normalized)
    }
}

impl Serialize for PyNormalizer {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.normalizer.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PyNormalizer {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(PyNormalizer::new(Arc::deserialize(deserializer)?))
    }
}

#[pymethods]
impl PyNormalizer {
    fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        let data = serde_json::to_string(&self.normalizer).map_err(|e| {
            exceptions::Exception::py_err(format!(
                "Error while attempting to pickle Normalizer: {}",
                e.to_string()
            ))
        })?;
        Ok(PyBytes::new(py, data.as_bytes()).to_object(py))
    }

    fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        match state.extract::<&PyBytes>(py) {
            Ok(s) => {
                self.normalizer = serde_json::from_slice(s.as_bytes()).map_err(|e| {
                    exceptions::Exception::py_err(format!(
                        "Error while attempting to unpickle Normalizer: {}",
                        e.to_string()
                    ))
                })?;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }
}

#[pyclass(extends=PyNormalizer, module = "tokenizers.normalizers", name=BertNormalizer)]
pub struct PyBertNormalizer {}
#[pymethods]
impl PyBertNormalizer {
    #[new]
    #[args(kwargs = "**")]
    fn new(kwargs: Option<&PyDict>) -> PyResult<(Self, PyNormalizer)> {
        let mut clean_text = true;
        let mut handle_chinese_chars = true;
        let mut strip_accents = None;
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
        let normalizer =
            BertNormalizer::new(clean_text, handle_chinese_chars, strip_accents, lowercase);
        Ok((PyBertNormalizer {}, PyNormalizer::new(Arc::new(normalizer))))
    }
}

#[pyclass(extends=PyNormalizer, module = "tokenizers.normalizers", name=NFD)]
pub struct PyNFD {}
#[pymethods]
impl PyNFD {
    #[new]
    fn new() -> PyResult<(Self, PyNormalizer)> {
        Ok((PyNFD {}, PyNormalizer::new(Arc::new(NFD))))
    }
}

#[pyclass(extends=PyNormalizer, module = "tokenizers.normalizers", name=NFKD)]
pub struct PyNFKD {}
#[pymethods]
impl PyNFKD {
    #[new]
    fn new() -> PyResult<(Self, PyNormalizer)> {
        Ok((PyNFKD {}, PyNormalizer::new(Arc::new(NFKD))))
    }
}

#[pyclass(extends=PyNormalizer, module = "tokenizers.normalizers", name=NFC)]
pub struct PyNFC {}
#[pymethods]
impl PyNFC {
    #[new]
    fn new() -> PyResult<(Self, PyNormalizer)> {
        Ok((PyNFC {}, PyNormalizer::new(Arc::new(NFC))))
    }
}

#[pyclass(extends=PyNormalizer, module = "tokenizers.normalizers", name=NFKC)]
pub struct PyNFKC {}
#[pymethods]
impl PyNFKC {
    #[new]
    fn new() -> PyResult<(Self, PyNormalizer)> {
        Ok((PyNFKC {}, PyNormalizer::new(Arc::new(NFKC))))
    }
}

#[pyclass(extends=PyNormalizer, module = "tokenizers.normalizers", name=Sequence)]
pub struct PySequence {}
#[pymethods]
impl PySequence {
    #[new]
    fn new(normalizers: &PyList) -> PyResult<(Self, PyNormalizer)> {
        let normalizers = normalizers
            .iter()
            .map(|n| {
                let normalizer: PyRef<PyNormalizer> = n.extract()?;
                let normalizer = PyNormalizer::new(normalizer.normalizer.clone());
                let boxed = Box::new(normalizer);
                Ok(boxed as Box<dyn Normalizer>)
            })
            .collect::<PyResult<_>>()?;

        Ok((
            PySequence {},
            PyNormalizer::new(Arc::new(Sequence::new(normalizers))),
        ))
    }

    fn __getnewargs__<'p>(&self, py: Python<'p>) -> PyResult<&'p PyTuple> {
        Ok(PyTuple::new(py, &[PyList::empty(py)]))
    }
}

#[pyclass(extends=PyNormalizer, module = "tokenizers.normalizers", name=Lowercase)]
pub struct PyLowercase {}
#[pymethods]
impl PyLowercase {
    #[new]
    fn new() -> PyResult<(Self, PyNormalizer)> {
        Ok((PyLowercase {}, PyNormalizer::new(Arc::new(Lowercase))))
    }
}

#[pyclass(extends=PyNormalizer, module = "tokenizers.normalizers", name=Strip)]
pub struct PyStrip {}
#[pymethods]
impl PyStrip {
    #[new]
    #[args(kwargs = "**")]
    fn new(kwargs: Option<&PyDict>) -> PyResult<(Self, PyNormalizer)> {
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
            PyStrip {},
            PyNormalizer::new(Arc::new(Strip::new(left, right))),
        ))
    }
}
