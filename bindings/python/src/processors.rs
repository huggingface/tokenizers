use std::sync::Arc;

use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::*;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use tk::processors::bert::BertProcessing;
use tk::processors::byte_level::ByteLevel;
use tk::processors::roberta::RobertaProcessing;
use tk::{Encoding, PostProcessor};
use tokenizers as tk;

#[pyclass(dict, module = "tokenizers.processors", name=PostProcessor)]
#[derive(Clone)]
pub struct PyPostProcessor {
    pub processor: Arc<dyn PostProcessor>,
}

impl PyPostProcessor {
    pub fn new(processor: Arc<dyn PostProcessor>) -> Self {
        PyPostProcessor { processor }
    }
}

#[typetag::serde]
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

impl Serialize for PyPostProcessor {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.processor.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PyPostProcessor {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(PyPostProcessor::new(Arc::deserialize(deserializer)?))
    }
}

#[pymethods]
impl PyPostProcessor {
    fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        let data = serde_json::to_string(self.processor.as_ref()).map_err(|e| {
            exceptions::Exception::py_err(format!(
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
                    exceptions::Exception::py_err(format!(
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
}

#[pyclass(extends=PyPostProcessor, module = "tokenizers.processors", name=BertProcessing)]
pub struct PyBertProcessing {}
#[pymethods]
impl PyBertProcessing {
    #[new]
    fn new(sep: (String, u32), cls: (String, u32)) -> PyResult<(Self, PyPostProcessor)> {
        Ok((
            PyBertProcessing {},
            PyPostProcessor::new(Arc::new(BertProcessing::new(sep, cls))),
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
        Ok((PyRobertaProcessing {}, PyPostProcessor::new(Arc::new(proc))))
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
        Ok((PyByteLevel {}, PyPostProcessor::new(Arc::new(byte_level))))
    }
}
