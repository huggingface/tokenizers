extern crate tokenizers as tk;

use super::utils::Container;
use pyo3::prelude::*;
use pyo3::types::*;

#[pyclass(dict)]
pub struct PostProcessor {
    pub processor: Container<dyn tk::tokenizer::PostProcessor + Sync>,
}

#[pymethods]
impl PostProcessor {
    fn num_special_tokens_to_add(&self, is_pair: bool) -> usize {
        self.processor.execute(|p| p.added_tokens(is_pair))
    }
}

#[pyclass(extends=PostProcessor)]
pub struct BertProcessing {}
#[pymethods]
impl BertProcessing {
    #[new]
    fn new(obj: &PyRawObject, sep: (String, u32), cls: (String, u32)) -> PyResult<()> {
        Ok(obj.init(PostProcessor {
            processor: Container::Owned(Box::new(tk::processors::bert::BertProcessing::new(
                sep, cls,
            ))),
        }))
    }
}

#[pyclass(extends=PostProcessor)]
pub struct RobertaProcessing {}
#[pymethods]
impl RobertaProcessing {
    #[new]
    fn new(obj: &PyRawObject, sep: (String, u32), cls: (String, u32)) -> PyResult<()> {
        Ok(obj.init(PostProcessor {
            processor: Container::Owned(Box::new(tk::processors::roberta::RobertaProcessing::new(
                sep, cls,
            ))),
        }))
    }
}

#[pyclass(extends=PostProcessor)]
pub struct ByteLevel {}
#[pymethods]
impl ByteLevel {
    #[new]
    #[args(kwargs = "**")]
    fn new(obj: &PyRawObject, kwargs: Option<&PyDict>) -> PyResult<()> {
        let mut byte_level = tk::processors::byte_level::ByteLevel::default();

        if let Some(kwargs) = kwargs {
            for (key, value) in kwargs {
                let key: &str = key.extract()?;
                match key {
                    "trim_offsets" => byte_level = byte_level.trim_offsets(value.extract()?),
                    _ => println!("Ignored unknown kwargs option {}", key),
                }
            }
        }
        Ok(obj.init(PostProcessor {
            processor: Container::Owned(Box::new(byte_level)),
        }))
    }
}
