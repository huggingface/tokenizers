extern crate tokenizers as tk;

use super::utils::Container;

use pyo3::exceptions;
use pyo3::prelude::*;

/// Represents any Model to be used with a Tokenizer
/// This class is to be constructed from specific models
#[pyclass]
pub struct Model {
    pub model: Container<dyn tk::tokenizer::Model + Sync>,
}

#[pymethods]
impl Model {
    #[new]
    fn new(_obj: &PyRawObject) -> PyResult<Self> {
        Err(exceptions::Exception::py_err(
            "Cannot create a Model directly",
        ))
    }
}

/// BPE Model
/// Allows the creation of a BPE Model to be used with a Tokenizer
#[pyclass]
pub struct BPE {}

#[pymethods]
impl BPE {
    #[staticmethod]
    fn from_files(vocab: &str, merges: &str) -> PyResult<Model> {
        match tk::models::bpe::BPE::from_files(vocab, merges) {
            Err(e) => {
                println!("Error: {:?}", e);
                Err(exceptions::Exception::py_err(
                    "Error while initializing BPE",
                ))
            }
            Ok(bpe) => Ok(Model {
                model: Container::Owned(Box::new(bpe)),
            }),
        }
    }
}
