use std::sync::Arc;

use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::*;

use crate::error::ToPyResult;
use serde::ser::SerializeStruct;
use serde::{Deserialize, Serialize, Serializer};
use tk::normalizers::{
    BertNormalizer, Lowercase, Nmt, NormalizerWrapper, Precompiled, Replace, Strip, StripAccents,
    NFC, NFD, NFKC, NFKD,
};
use tk::{NormalizedString, Normalizer};
use tokenizers as tk;

#[pyclass(dict, module = "tokenizers.normalizers", name=Normalizer)]
#[derive(Clone, Serialize, Deserialize)]
pub struct PyNormalizer {
    #[serde(flatten)]
    pub(crate) normalizer: PyNormalizerWrapper,
}

impl PyNormalizer {
    pub(crate) fn new(normalizer: PyNormalizerWrapper) -> Self {
        PyNormalizer { normalizer }
    }
    pub(crate) fn get_as_subtype(&self) -> PyResult<PyObject> {
        let base = self.clone();
        let gil = Python::acquire_gil();
        let py = gil.python();
        match self.normalizer {
            PyNormalizerWrapper::Sequence(_) => Py::new(py, (PySequence {}, base)).map(Into::into),
            PyNormalizerWrapper::Wrapped(ref inner) => match inner.as_ref() {
                NormalizerWrapper::Sequence(_) => {
                    Py::new(py, (PySequence {}, base)).map(Into::into)
                }
                NormalizerWrapper::BertNormalizer(_) => {
                    Py::new(py, (PyBertNormalizer {}, base)).map(Into::into)
                }
                NormalizerWrapper::StripNormalizer(_) => {
                    Py::new(py, (PyBertNormalizer {}, base)).map(Into::into)
                }
                NormalizerWrapper::StripAccents(_) => {
                    Py::new(py, (PyStripAccents {}, base)).map(Into::into)
                }
                NormalizerWrapper::NFC(_) => Py::new(py, (PyNFC {}, base)).map(Into::into),
                NormalizerWrapper::NFD(_) => Py::new(py, (PyNFD {}, base)).map(Into::into),
                NormalizerWrapper::NFKC(_) => Py::new(py, (PyNFKC {}, base)).map(Into::into),
                NormalizerWrapper::NFKD(_) => Py::new(py, (PyNFKD {}, base)).map(Into::into),
                NormalizerWrapper::Lowercase(_) => {
                    Py::new(py, (PyLowercase {}, base)).map(Into::into)
                }
                NormalizerWrapper::Precompiled(_) => {
                    Py::new(py, (PyPrecompiled {}, base)).map(Into::into)
                }
                NormalizerWrapper::Replace(_) => Py::new(py, (PyReplace {}, base)).map(Into::into),
                NormalizerWrapper::Nmt(_) => Py::new(py, (PyNmt {}, base)).map(Into::into),
            },
        }
    }
}

impl Normalizer for PyNormalizer {
    fn normalize(&self, normalized: &mut NormalizedString) -> tk::Result<()> {
        self.normalizer.normalize(normalized)
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

    fn normalize_str(&self, sequence: &str) -> PyResult<String> {
        let mut normalized = NormalizedString::from(sequence);
        ToPyResult(self.normalizer.normalize(&mut normalized)).into_py()?;
        Ok(normalized.get().to_owned())
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
        Ok((PyBertNormalizer {}, normalizer.into()))
    }
}

#[pyclass(extends=PyNormalizer, module = "tokenizers.normalizers", name=NFD)]
pub struct PyNFD {}
#[pymethods]
impl PyNFD {
    #[new]
    fn new() -> PyResult<(Self, PyNormalizer)> {
        Ok((PyNFD {}, PyNormalizer::new(NFD.into())))
    }
}

#[pyclass(extends=PyNormalizer, module = "tokenizers.normalizers", name=NFKD)]
pub struct PyNFKD {}
#[pymethods]
impl PyNFKD {
    #[new]
    fn new() -> PyResult<(Self, PyNormalizer)> {
        Ok((PyNFKD {}, NFKD.into()))
    }
}

#[pyclass(extends=PyNormalizer, module = "tokenizers.normalizers", name=NFC)]
pub struct PyNFC {}
#[pymethods]
impl PyNFC {
    #[new]
    fn new() -> PyResult<(Self, PyNormalizer)> {
        Ok((PyNFC {}, NFC.into()))
    }
}

#[pyclass(extends=PyNormalizer, module = "tokenizers.normalizers", name=NFKC)]
pub struct PyNFKC {}
#[pymethods]
impl PyNFKC {
    #[new]
    fn new() -> PyResult<(Self, PyNormalizer)> {
        Ok((PyNFKC {}, NFKC.into()))
    }
}

#[pyclass(extends=PyNormalizer, module = "tokenizers.normalizers", name=Sequence)]
pub struct PySequence {}
#[pymethods]
impl PySequence {
    #[new]
    fn new(normalizers: &PyList) -> PyResult<(Self, PyNormalizer)> {
        let mut sequence = Vec::with_capacity(normalizers.len());
        for n in normalizers.iter() {
            let normalizer: PyRef<PyNormalizer> = n.extract()?;
            match &normalizer.normalizer {
                PyNormalizerWrapper::Sequence(inner) => sequence.extend(inner.iter().cloned()),
                PyNormalizerWrapper::Wrapped(inner) => sequence.push(inner.clone()),
            }
        }
        Ok((
            PySequence {},
            PyNormalizer::new(PyNormalizerWrapper::Sequence(sequence)),
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
        Ok((PyLowercase {}, Lowercase.into()))
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

        Ok((PyStrip {}, Strip::new(left, right).into()))
    }
}

#[pyclass(extends=PyNormalizer, module = "tokenizers.normalizers", name=StripAccents)]
pub struct PyStripAccents {}
#[pymethods]
impl PyStripAccents {
    #[new]
    fn new() -> PyResult<(Self, PyNormalizer)> {
        Ok((PyStripAccents {}, StripAccents.into()))
    }
}

#[derive(Clone, Deserialize)]
#[serde(untagged)]
pub(crate) enum PyNormalizerWrapper {
    Sequence(Vec<Arc<NormalizerWrapper>>),
    Wrapped(Arc<NormalizerWrapper>),
}

impl Serialize for PyNormalizerWrapper {
    fn serialize<S>(&self, serializer: S) -> Result<<S as Serializer>::Ok, <S as Serializer>::Error>
    where
        S: Serializer,
    {
        match self {
            PyNormalizerWrapper::Sequence(seq) => {
                let mut ser = serializer.serialize_struct("Sequence", 2)?;
                ser.serialize_field("type", "Sequence")?;
                ser.serialize_field("normalizers", seq)?;
                ser.end()
            }
            PyNormalizerWrapper::Wrapped(inner) => inner.serialize(serializer),
        }
    }
}

impl<I> From<I> for PyNormalizerWrapper
where
    I: Into<NormalizerWrapper>,
{
    fn from(norm: I) -> Self {
        PyNormalizerWrapper::Wrapped(Arc::new(norm.into()))
    }
}

impl<I> From<I> for PyNormalizer
where
    I: Into<NormalizerWrapper>,
{
    fn from(norm: I) -> Self {
        PyNormalizer {
            normalizer: norm.into().into(),
        }
    }
}

impl Normalizer for PyNormalizerWrapper {
    fn normalize(&self, normalized: &mut NormalizedString) -> tk::Result<()> {
        match self {
            PyNormalizerWrapper::Wrapped(inner) => inner.normalize(normalized),
            PyNormalizerWrapper::Sequence(inner) => {
                inner.iter().map(|n| n.normalize(normalized)).collect()
            }
        }
    }
}

#[pyclass(extends=PyNormalizer, module = "tokenizers.normalizers", name=Nmt)]
pub struct PyNmt {}
#[pymethods]
impl PyNmt {
    #[new]
    fn new() -> PyResult<(Self, PyNormalizer)> {
        Ok((PyNmt {}, Nmt.into()))
    }
}

#[pyclass(extends=PyNormalizer, module = "tokenizers.normalizers", name=Precompiled)]
pub struct PyPrecompiled {}
#[pymethods]
impl PyPrecompiled {
    #[new]
    fn new(py_precompiled_charsmap: &PyBytes) -> PyResult<(Self, PyNormalizer)> {
        let precompiled_charsmap: &[u8] = FromPyObject::extract(py_precompiled_charsmap)?;
        Ok((
            PyPrecompiled {},
            Precompiled::from(precompiled_charsmap)
                .map_err(|e| {
                    exceptions::Exception::py_err(format!(
                        "Error while attempting to build Precompiled normalizer: {}",
                        e.to_string()
                    ))
                })?
                .into(),
        ))
    }
}

#[pyclass(extends=PyNormalizer, module = "tokenizers.normalizers", name=Replace)]
pub struct PyReplace {}
#[pymethods]
impl PyReplace {
    #[new]
    fn new(pattern: String, content: String) -> PyResult<(Self, PyNormalizer)> {
        Ok((PyReplace {}, Replace::new(pattern, content).into()))
    }
}

#[cfg(test)]
mod test {
    use pyo3::{AsPyRef, Python};
    use tk::normalizers::unicode::{NFC, NFKC};
    use tk::normalizers::utils::Sequence;
    use tk::normalizers::NormalizerWrapper;

    use crate::normalizers::{PyNormalizer, PyNormalizerWrapper};

    #[test]
    fn get_subtype() {
        let py_norm = PyNormalizer::new(NFC.into());
        let py_nfc = py_norm.get_as_subtype().unwrap();
        let gil = Python::acquire_gil();
        assert_eq!(
            "tokenizers.normalizers.NFC",
            py_nfc.as_ref(gil.python()).get_type().name()
        );
    }

    #[test]
    fn serialize() {
        let py_wrapped: PyNormalizerWrapper = NFKC.into();
        let py_ser = serde_json::to_string(&py_wrapped).unwrap();
        let rs_wrapped = NormalizerWrapper::NFKC(NFKC);
        let rs_ser = serde_json::to_string(&rs_wrapped).unwrap();
        assert_eq!(py_ser, rs_ser);
        let py_norm: PyNormalizer = serde_json::from_str(&rs_ser).unwrap();
        match py_norm.normalizer {
            PyNormalizerWrapper::Wrapped(nfc) => match nfc.as_ref() {
                NormalizerWrapper::NFKC(_) => {}
                _ => panic!("Expected NFKC"),
            },
            _ => panic!("Expected wrapped, not sequence."),
        }

        let py_seq: PyNormalizerWrapper = Sequence::new(vec![NFC.into(), NFKC.into()]).into();
        let py_wrapper_ser = serde_json::to_string(&py_seq).unwrap();
        let rs_wrapped = NormalizerWrapper::Sequence(Sequence::new(vec![NFC.into(), NFKC.into()]));
        let rs_ser = serde_json::to_string(&rs_wrapped).unwrap();
        assert_eq!(py_wrapper_ser, rs_ser);

        let py_seq = PyNormalizer::new(py_seq);
        let py_ser = serde_json::to_string(&py_seq).unwrap();
        assert_eq!(py_wrapper_ser, py_ser);

        let rs_seq = Sequence::new(vec![NFC.into(), NFKC.into()]);
        let rs_ser = serde_json::to_string(&rs_seq).unwrap();
        assert_eq!(py_wrapper_ser, rs_ser);
    }
}
