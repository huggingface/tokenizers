use std::convert::TryInto;
use std::sync::Arc;

use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::*;

use crate::encoding::PyEncoding;
use crate::error::ToPyResult;
use serde::{Deserialize, Serialize};
use tk::processors::bert::BertProcessing;
use tk::processors::byte_level::ByteLevel;
use tk::processors::roberta::RobertaProcessing;
use tk::processors::template::{SpecialToken, Template};
use tk::processors::PostProcessorWrapper;
use tk::{Encoding, PostProcessor};
use tokenizers as tk;

#[pyclass(dict, module = "tokenizers.processors", name=PostProcessor)]
#[derive(Clone, Deserialize, Serialize)]
pub struct PyPostProcessor {
    #[serde(flatten)]
    pub processor: Arc<PostProcessorWrapper>,
}

impl PyPostProcessor {
    pub fn new(processor: Arc<PostProcessorWrapper>) -> Self {
        PyPostProcessor { processor }
    }

    pub(crate) fn get_as_subtype(&self) -> PyResult<PyObject> {
        let base = self.clone();
        let gil = Python::acquire_gil();
        let py = gil.python();
        Ok(match self.processor.as_ref() {
            PostProcessorWrapper::ByteLevel(_) => Py::new(py, (PyByteLevel {}, base))?.into_py(py),
            PostProcessorWrapper::Bert(_) => Py::new(py, (PyBertProcessing {}, base))?.into_py(py),
            PostProcessorWrapper::Roberta(_) => {
                Py::new(py, (PyRobertaProcessing {}, base))?.into_py(py)
            }
            PostProcessorWrapper::Template(_) => {
                Py::new(py, (PyTemplateProcessing {}, base))?.into_py(py)
            }
        })
    }
}

impl PostProcessor for PyPostProcessor {
    fn added_tokens(&self, is_pair: bool) -> usize {
        self.processor.added_tokens(is_pair)
    }

    fn process(
        &self,
        encoding: Encoding,
        pair_encoding: Option<Encoding>,
        add_special_tokens: bool,
    ) -> tk::Result<Encoding> {
        self.processor
            .process(encoding, pair_encoding, add_special_tokens)
    }
}

#[pymethods]
impl PyPostProcessor {
    fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        let data = serde_json::to_string(self.processor.as_ref()).map_err(|e| {
            exceptions::PyException::new_err(format!(
                "Error while attempting to pickle PostProcessor: {}",
                e.to_string()
            ))
        })?;
        Ok(PyBytes::new(py, data.as_bytes()).to_object(py))
    }

    fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        match state.extract::<&PyBytes>(py) {
            Ok(s) => {
                self.processor = serde_json::from_slice(s.as_bytes()).map_err(|e| {
                    exceptions::PyException::new_err(format!(
                        "Error while attempting to unpickle PostProcessor: {}",
                        e.to_string()
                    ))
                })?;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    fn num_special_tokens_to_add(&self, is_pair: bool) -> usize {
        self.processor.added_tokens(is_pair)
    }

    #[args(pair = "None", add_special_tokens = "true")]
    fn process(
        &self,
        encoding: &PyEncoding,
        pair: Option<&PyEncoding>,
        add_special_tokens: bool,
    ) -> PyResult<PyEncoding> {
        let final_encoding = ToPyResult(self.processor.process(
            encoding.encoding.clone(),
            pair.map(|e| e.encoding.clone()),
            add_special_tokens,
        ))
        .into_py()?;
        Ok(final_encoding.into())
    }
}

#[pyclass(extends=PyPostProcessor, module = "tokenizers.processors", name=BertProcessing)]
pub struct PyBertProcessing {}
#[pymethods]
impl PyBertProcessing {
    #[new]
    fn new(sep: (String, u32), cls: (String, u32)) -> PyResult<(Self, PyPostProcessor)> {
        Ok((
            PyBertProcessing {},
            PyPostProcessor::new(Arc::new(BertProcessing::new(sep, cls).into())),
        ))
    }

    fn __getnewargs__<'p>(&self, py: Python<'p>) -> PyResult<&'p PyTuple> {
        Ok(PyTuple::new(py, &[("", 0), ("", 0)]))
    }
}

#[pyclass(extends=PyPostProcessor, module = "tokenizers.processors", name=RobertaProcessing)]
pub struct PyRobertaProcessing {}
#[pymethods]
impl PyRobertaProcessing {
    #[new]
    #[args(trim_offsets = true, add_prefix_space = true)]
    fn new(
        sep: (String, u32),
        cls: (String, u32),
        trim_offsets: bool,
        add_prefix_space: bool,
    ) -> PyResult<(Self, PyPostProcessor)> {
        let proc = RobertaProcessing::new(sep, cls)
            .trim_offsets(trim_offsets)
            .add_prefix_space(add_prefix_space);
        Ok((
            PyRobertaProcessing {},
            PyPostProcessor::new(Arc::new(proc.into())),
        ))
    }

    fn __getnewargs__<'p>(&self, py: Python<'p>) -> PyResult<&'p PyTuple> {
        Ok(PyTuple::new(py, &[("", 0), ("", 0)]))
    }
}

#[pyclass(extends=PyPostProcessor, module = "tokenizers.processors", name=ByteLevel)]
pub struct PyByteLevel {}
#[pymethods]
impl PyByteLevel {
    #[new]
    #[args(kwargs = "**")]
    fn new(kwargs: Option<&PyDict>) -> PyResult<(Self, PyPostProcessor)> {
        let mut byte_level = ByteLevel::default();

        if let Some(kwargs) = kwargs {
            for (key, value) in kwargs {
                let key: &str = key.extract()?;
                match key {
                    "trim_offsets" => byte_level = byte_level.trim_offsets(value.extract()?),
                    _ => println!("Ignored unknown kwargs option {}", key),
                }
            }
        }
        Ok((
            PyByteLevel {},
            PyPostProcessor::new(Arc::new(byte_level.into())),
        ))
    }
}

#[derive(Clone, Debug)]
pub struct PySpecialToken(SpecialToken);

impl From<PySpecialToken> for SpecialToken {
    fn from(v: PySpecialToken) -> Self {
        v.0
    }
}

impl FromPyObject<'_> for PySpecialToken {
    fn extract(ob: &PyAny) -> PyResult<Self> {
        if let Ok(v) = ob.extract::<(String, u32)>() {
            Ok(Self(v.into()))
        } else if let Ok(v) = ob.extract::<(u32, String)>() {
            Ok(Self(v.into()))
        } else if let Ok(d) = ob.downcast::<PyDict>() {
            let id = d
                .get_item("id")
                .ok_or_else(|| exceptions::PyValueError::new_err("`id` must be specified"))?
                .extract::<String>()?;
            let ids = d
                .get_item("ids")
                .ok_or_else(|| exceptions::PyValueError::new_err("`ids` must be specified"))?
                .extract::<Vec<u32>>()?;
            let tokens = d
                .get_item("tokens")
                .ok_or_else(|| exceptions::PyValueError::new_err("`tokens` must be specified"))?
                .extract::<Vec<String>>()?;

            Ok(Self(
                ToPyResult(SpecialToken::new(id, ids, tokens)).into_py()?,
            ))
        } else {
            Err(exceptions::PyTypeError::new_err(
                "Expected Union[Tuple[str, int], Tuple[int, str], dict]",
            ))
        }
    }
}

#[derive(Clone, Debug)]
pub struct PyTemplate(Template);

impl From<PyTemplate> for Template {
    fn from(v: PyTemplate) -> Self {
        v.0
    }
}

impl FromPyObject<'_> for PyTemplate {
    fn extract(ob: &PyAny) -> PyResult<Self> {
        if let Ok(s) = ob.extract::<&str>() {
            Ok(Self(
                s.try_into().map_err(exceptions::PyValueError::new_err)?,
            ))
        } else if let Ok(s) = ob.extract::<Vec<&str>>() {
            Ok(Self(
                s.try_into().map_err(exceptions::PyValueError::new_err)?,
            ))
        } else {
            Err(exceptions::PyTypeError::new_err(
                "Expected Union[str, List[str]]",
            ))
        }
    }
}

#[pyclass(extends=PyPostProcessor, module = "tokenizers.processors", name=TemplateProcessing)]
pub struct PyTemplateProcessing {}
#[pymethods]
impl PyTemplateProcessing {
    #[new]
    #[args(single = "None", pair = "None", special_tokens = "None")]
    fn new(
        single: Option<PyTemplate>,
        pair: Option<PyTemplate>,
        special_tokens: Option<Vec<PySpecialToken>>,
    ) -> PyResult<(Self, PyPostProcessor)> {
        let mut builder = tk::processors::template::TemplateProcessing::builder();

        if let Some(seq) = single {
            builder.single(seq.into());
        }
        if let Some(seq) = pair {
            builder.pair(seq.into());
        }
        if let Some(sp) = special_tokens {
            builder.special_tokens(sp);
        }
        let processor = builder.build().map_err(exceptions::PyValueError::new_err)?;

        Ok((
            PyTemplateProcessing {},
            PyPostProcessor::new(Arc::new(processor.into())),
        ))
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use pyo3::prelude::*;
    use tk::processors::bert::BertProcessing;
    use tk::processors::PostProcessorWrapper;

    use crate::processors::PyPostProcessor;

    #[test]
    fn get_subtype() {
        let py_proc = PyPostProcessor::new(Arc::new(
            BertProcessing::new(("SEP".into(), 0), ("CLS".into(), 1)).into(),
        ));
        let py_bert = py_proc.get_as_subtype().unwrap();
        let gil = Python::acquire_gil();
        assert_eq!(
            "tokenizers.processors.BertProcessing",
            py_bert.as_ref(gil.python()).get_type().name()
        );
    }

    #[test]
    fn serialize() {
        let rs_processing = BertProcessing::new(("SEP".into(), 0), ("CLS".into(), 1));
        let rs_wrapper: PostProcessorWrapper = rs_processing.clone().into();
        let rs_processing_ser = serde_json::to_string(&rs_processing).unwrap();
        let rs_wrapper_ser = serde_json::to_string(&rs_wrapper).unwrap();

        let py_processing = PyPostProcessor::new(Arc::new(rs_wrapper));
        let py_ser = serde_json::to_string(&py_processing).unwrap();
        assert_eq!(py_ser, rs_processing_ser);
        assert_eq!(py_ser, rs_wrapper_ser);

        let py_processing: PyPostProcessor = serde_json::from_str(&rs_processing_ser).unwrap();
        match py_processing.processor.as_ref() {
            PostProcessorWrapper::Bert(_) => (),
            _ => panic!("Expected Bert postprocessor."),
        }

        let py_processing: PyPostProcessor = serde_json::from_str(&rs_wrapper_ser).unwrap();
        match py_processing.processor.as_ref() {
            PostProcessorWrapper::Bert(_) => (),
            _ => panic!("Expected Bert postprocessor."),
        }
    }
}
