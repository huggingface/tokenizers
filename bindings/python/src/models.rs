extern crate tokenizers as tk;

use super::error::ToPyResult;
use super::utils::Container;
use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::*;
use std::path::Path;

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

    fn save(&self, folder: &str, name: &str) -> PyResult<Vec<String>> {
        let saved: PyResult<Vec<_>> = ToPyResult(
            self.model
                .execute(|model| model.save(Path::new(folder), name)),
        )
        .into();

        Ok(saved?
            .into_iter()
            .map(|path| path.to_string_lossy().into_owned())
            .collect())
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
    #[args(kwargs = "**")]
    fn from_files(vocab: &str, merges: &str, kwargs: Option<&PyDict>) -> PyResult<Model> {
        let builder: PyResult<_> =
            ToPyResult(tk::models::bpe::BPE::from_files(vocab, merges)).into();
        let mut builder = builder?;

        if let Some(kwargs) = kwargs {
            for (key, value) in kwargs {
                let key: &str = key.extract()?;
                match key {
                    "cache_capacity" => builder = builder.cache_capacity(value.extract()?),
                    "dropout" => {
                        if let Some(dropout) = value.extract()? {
                            builder = builder.dropout(dropout);
                        }
                    }
                    "unk_token" => {
                        if let Some(unk) = value.extract()? {
                            builder = builder.unk_token(unk);
                        }
                    }
                    "continuing_subword_prefix" => {
                        builder = builder.continuing_subword_prefix(value.extract()?)
                    }
                    "end_of_word_suffix" => builder = builder.end_of_word_suffix(value.extract()?),
                    _ => println!("Ignored unknown kwarg option {}", key),
                };
            }
        }

        match builder.build() {
            Err(e) => Err(exceptions::Exception::py_err(format!(
                "Error while initializing BPE: {}",
                e
            ))),
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
            model: Container::Owned(Box::new(tk::models::bpe::BPE::default())),
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
    #[args(kwargs = "**")]
    fn from_files(vocab: &str, kwargs: Option<&PyDict>) -> PyResult<Model> {
        let mut unk_token = String::from("[UNK]");
        let mut max_input_chars_per_word = Some(100);

        if let Some(kwargs) = kwargs {
            for (key, val) in kwargs {
                let key: &str = key.extract()?;
                match key {
                    "unk_token" => unk_token = val.extract()?,
                    "max_input_chars_per_word" => max_input_chars_per_word = Some(val.extract()?),
                    _ => println!("Ignored unknown kwargs option {}", key),
                }
            }
        }

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

    #[staticmethod]
    fn empty() -> Model {
        Model {
            model: Container::Owned(Box::new(tk::models::wordpiece::WordPiece::default())),
        }
    }
}

#[pyclass]
pub struct WordLevel {}

#[pymethods]
impl WordLevel {
    #[staticmethod]
    #[args(kwargs = "**")]
    fn from_files(vocab: &str, kwargs: Option<&PyDict>) -> PyResult<Model> {
        let mut unk_token = String::from("<unk>");

        if let Some(kwargs) = kwargs {
            for (key, val) in kwargs {
                let key: &str = key.extract()?;
                match key {
                    "unk_token" => unk_token = val.extract()?,
                    _ => println!("Ignored unknown kwargs option {}", key),
                }
            }
        }

        match tk::models::wordlevel::WordLevel::from_files(
            vocab,
            unk_token,
        ) {
            Err(e) => {
                println!("Errors: {:?}", e);
                Err(exceptions::Exception::py_err(
                    "Error while initializing WordLevel",
                ))
            }
            Ok(model) => Ok(Model {
                model: Container::Owned(Box::new(model)),
            }),
        }
    }
}
