use std::collections::hash_map::RandomState;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::*;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use tk::models::bpe::BPE;
use tk::models::wordlevel::WordLevel;
use tk::models::wordpiece::WordPiece;
use tk::{Model, Token};
use tokenizers as tk;

use super::error::ToPyResult;
use tk::models::ModelWrapper;

/// A Model represents some tokenization algorithm like BPE or Word
/// This class cannot be constructed directly. Please use one of the concrete models.
#[pyclass(module = "tokenizers.models", name=Model)]
#[derive(Clone)]
pub struct PyModel {
    pub model: Arc<ModelWrapper>,
}

impl PyModel {
    pub(crate) fn new(model: Arc<ModelWrapper>) -> Self {
        PyModel { model }
    }
}

impl Serialize for PyModel {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.model.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PyModel {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(PyModel {
            model: Arc::deserialize(deserializer)?,
        })
    }
}

#[typetag::serde]
impl Model for PyModel {
    fn tokenize(&self, tokens: &str) -> tk::Result<Vec<Token>> {
        self.model.tokenize(tokens)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.model.token_to_id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<&str> {
        self.model.id_to_token(id)
    }

    fn get_vocab(&self) -> &HashMap<String, u32, RandomState> {
        self.model.get_vocab()
    }

    fn get_vocab_size(&self) -> usize {
        self.model.get_vocab_size()
    }

    fn save(&self, folder: &Path, name: Option<&str>) -> tk::Result<Vec<PathBuf>> {
        self.model.save(folder, name)
    }
}

#[pymethods]
impl PyModel {
    #[new]
    fn __new__() -> PyResult<Self> {
        // Instantiate a default empty model. This doesn't really make sense, but we need
        // to be able to instantiate an empty model for pickle capabilities.
        Ok(PyModel {
            model: Arc::new(BPE::default().into()),
        })
    }

    fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        let data = serde_json::to_string(&self.model).map_err(|e| {
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
                self.model = serde_json::from_slice(s.as_bytes()).map_err(|e| {
                    exceptions::Exception::py_err(format!(
                        "Error while attempting to unpickle Model: {}",
                        e.to_string()
                    ))
                })?;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    fn save(&self, folder: &str, name: Option<&str>) -> PyResult<Vec<String>> {
        let saved: PyResult<Vec<_>> = ToPyResult(self.model.save(Path::new(folder), name)).into();

        Ok(saved?
            .into_iter()
            .map(|path| path.to_string_lossy().into_owned())
            .collect())
    }
}

/// BPE Model
/// Allows the creation of a BPE Model to be used with a Tokenizer
#[pyclass(extends=PyModel, module = "tokenizers.models", name=BPE)]
pub struct PyBPE {}

#[pymethods]
impl PyBPE {
    #[new]
    #[args(kwargs = "**")]
    fn new(
        vocab: Option<&str>,
        merges: Option<&str>,
        kwargs: Option<&PyDict>,
    ) -> PyResult<(Self, PyModel)> {
        if (vocab.is_some() && merges.is_none()) || (vocab.is_none() && merges.is_some()) {
            return Err(exceptions::ValueError::py_err(
                "`vocab` and `merges` must be both specified",
            ));
        }

        let mut builder = BPE::builder();
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
            Ok(bpe) => Ok((PyBPE {}, PyModel::new(Arc::new(bpe.into())))),
        }
    }
}

/// WordPiece Model
#[pyclass(extends=PyModel, module = "tokenizers.models", name=WordPiece)]
pub struct PyWordPiece {}

#[pymethods]
impl PyWordPiece {
    #[new]
    #[args(kwargs = "**")]
    fn new(vocab: Option<&str>, kwargs: Option<&PyDict>) -> PyResult<(Self, PyModel)> {
        let mut builder = WordPiece::builder();

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
            Ok(wordpiece) => Ok((PyWordPiece {}, PyModel::new(Arc::new(wordpiece.into())))),
        }
    }
}

#[pyclass(extends=PyModel, module = "tokenizers.models", name=WordLevel)]
pub struct PyWordLevel {}

#[pymethods]
impl PyWordLevel {
    #[new]
    #[args(kwargs = "**")]
    fn new(vocab: Option<&str>, kwargs: Option<&PyDict>) -> PyResult<(Self, PyModel)> {
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
            match WordLevel::from_files(vocab, unk_token) {
                Err(e) => {
                    println!("Errors: {:?}", e);
                    Err(exceptions::Exception::py_err(
                        "Error while initializing WordLevel",
                    ))
                }
                Ok(model) => Ok((PyWordLevel {}, PyModel::new(Arc::new(model.into())))),
            }
        } else {
            Ok((PyWordLevel {}, PyModel::new(Arc::new(WordLevel::default().into()))))
        }
    }
}
