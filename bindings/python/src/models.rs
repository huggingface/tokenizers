extern crate tokenizers as tk;

use super::error::ToPyResult;
use super::utils::Container;
use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::*;
use std::path::Path;

/// A Model represents some tokenization algorithm like BPE or Word
/// This class cannot be constructed directly. Please use one of the concrete models.
#[pyclass(module = "tokenizers.models")]
pub struct Model {
    pub model: Container<dyn tk::tokenizer::Model>,
}

#[pymethods]
impl Model {
    #[new]
    fn new() -> PyResult<Self> {
        // Instantiate a default empty model. This doesn't really make sense, but we need
        // to be able to instantiate an empty model for pickle capabilities.
        Ok(Model {
            model: Container::Owned(Box::new(tk::models::bpe::BPE::default())),
        })
    }

    fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        let data = self
            .model
            .execute(|model| serde_json::to_string(&model))
            .map_err(|e| {
                exceptions::Exception::py_err(format!(
                    "Error while attempting to pickle Model: {}",
                    e.to_string()
                ))
            })?;
        Ok(PyBytes::new(py, data.as_bytes()).to_object(py))
    }

    fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        match state.extract::<&PyBytes>(py) {
            Ok(s) => {
                self.model =
                    Container::Owned(serde_json::from_slice(s.as_bytes()).map_err(|e| {
                        exceptions::Exception::py_err(format!(
                            "Error while attempting to unpickle Model: {}",
                            e.to_string()
                        ))
                    })?);
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    fn save(&self, folder: &str, name: Option<&str>) -> PyResult<Vec<String>> {
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
#[pyclass(extends=Model, module = "tokenizers.models")]
pub struct BPE {}

#[pymethods]
impl BPE {
    #[new]
    #[args(kwargs = "**")]
    fn new(
        vocab: Option<&str>,
        merges: Option<&str>,
        kwargs: Option<&PyDict>,
    ) -> PyResult<(Self, Model)> {
        if (vocab.is_some() && merges.is_none()) || (vocab.is_none() && merges.is_some()) {
            return Err(exceptions::ValueError::py_err(
                "`vocab` and `merges` must be both specified",
            ));
        }

        let mut builder = tk::models::bpe::BPE::builder();
        if let (Some(vocab), Some(merges)) = (vocab, merges) {
            builder = builder.files(vocab.to_owned(), merges.to_owned());
        }
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
            Ok(bpe) => Ok((
                BPE {},
                Model {
                    model: Container::Owned(Box::new(bpe)),
                },
            )),
        }
    }
}

/// WordPiece Model
#[pyclass(extends=Model, module = "tokenizers.models")]
pub struct WordPiece {}

#[pymethods]
impl WordPiece {
    #[new]
    #[args(kwargs = "**")]
    fn new(vocab: Option<&str>, kwargs: Option<&PyDict>) -> PyResult<(Self, Model)> {
        let mut builder = tk::models::wordpiece::WordPiece::builder();

        if let Some(vocab) = vocab {
            builder = builder.files(vocab.to_owned());
        }

        if let Some(kwargs) = kwargs {
            for (key, val) in kwargs {
                let key: &str = key.extract()?;
                match key {
                    "unk_token" => {
                        builder = builder.unk_token(val.extract()?);
                    }
                    "max_input_chars_per_word" => {
                        builder = builder.max_input_chars_per_word(val.extract()?);
                    }
                    "continuing_subword_prefix" => {
                        builder = builder.continuing_subword_prefix(val.extract()?);
                    }
                    _ => println!("Ignored unknown kwargs option {}", key),
                }
            }
        }

        match builder.build() {
            Err(e) => {
                println!("Errors: {:?}", e);
                Err(exceptions::Exception::py_err(
                    "Error while initializing WordPiece",
                ))
            }
            Ok(wordpiece) => Ok((
                WordPiece {},
                Model {
                    model: Container::Owned(Box::new(wordpiece)),
                },
            )),
        }
    }
}

#[pyclass(extends=Model, module = "tokenizers.models")]
pub struct WordLevel {}

#[pymethods]
impl WordLevel {
    #[new]
    #[args(kwargs = "**")]
    fn new(vocab: Option<&str>, kwargs: Option<&PyDict>) -> PyResult<(Self, Model)> {
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

        if let Some(vocab) = vocab {
            match tk::models::wordlevel::WordLevel::from_files(vocab, unk_token) {
                Err(e) => {
                    println!("Errors: {:?}", e);
                    Err(exceptions::Exception::py_err(
                        "Error while initializing WordLevel",
                    ))
                }
                Ok(model) => Ok((
                    WordLevel {},
                    Model {
                        model: Container::Owned(Box::new(model)),
                    },
                )),
            }
        } else {
            Ok((
                WordLevel {},
                Model {
                    model: Container::Owned(Box::new(tk::models::wordlevel::WordLevel::default())),
                },
            ))
        }
    }
}
