extern crate tokenizers as tk;

use super::utils::Container;

use pyo3::exceptions;
use pyo3::prelude::*;

/// A Model represents some tokenization algorithm like BPE or Word
/// This class cannot be constructed directly. Please use one of the concrete models.
#[pyclass]
pub struct Model {
    pub model: Container<dyn tk::tokenizer::Model + Sync>,
}

#[pymethods]
impl Model {
    #[new]
    fn new(_obj: &PyRawObject) -> PyResult<()> {
        Err(exceptions::Exception::py_err(
            "Cannot create a Model directly. Use a concrete subclass",
        ))
    }
}

/// BPE Model
/// Allows the creation of a BPE Model to be used with a Tokenizer
#[pyclass]
pub struct BPE {}

#[pymethods]
impl BPE {
    /// from_files(vocab, merges, /)
    /// --
    ///
    /// Instanciate a new BPE model using the provided vocab and merges files
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

    /// empty()
    /// --
    ///
    /// Instanciate a new BPE model with empty vocab and merges
    #[staticmethod]
    fn empty() -> Model {
        Model {
            model: Container::Owned(Box::new(tk::models::bpe::BPE::empty())),
        }
    }
}

/// WordPiece Model
#[pyclass]
pub struct WordPiece {}

#[pymethods]
impl WordPiece {
    /// from_files(vocab, /)
    /// --
    ///
    /// Instantiate a new WordPiece model using the provided vocabulary file
    #[staticmethod]
    fn from_files(vocab: &str) -> PyResult<Model> {
        // TODO: Parse kwargs for these
        let unk_token = String::from("[UNK]");
        let max_input_chars_per_word = Some(100);

        match tk::models::wordpiece::WordPiece::from_files(
            vocab,
            unk_token,
            max_input_chars_per_word,
        ) {
            Err(e) => {
                println!("Errors: {:?}", e);
                Err(exceptions::Exception::py_err(
                    "Error while initializing WordPiece",
                ))
            }
            Ok(wordpiece) => Ok(Model {
                model: Container::Owned(Box::new(wordpiece)),
            }),
        }
    }
}
