use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use crate::token::PyToken;
use crate::trainers::PyTrainer;
use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::*;
use serde::{Deserialize, Serialize};
use tk::models::bpe::{BpeBuilder, Merges, Vocab, BPE};
use tk::models::unigram::Unigram;
use tk::models::wordlevel::WordLevel;
use tk::models::wordpiece::{WordPiece, WordPieceBuilder};
use tk::models::ModelWrapper;
use tk::{Model, Token};
use tokenizers as tk;

use super::error::{deprecation_warning, ToPyResult};

/// A Model represents some tokenization algorithm like BPE or Word
/// This class cannot be constructed directly. Please use one of the concrete models.
#[pyclass(module = "tokenizers.models", name=Model)]
#[derive(Clone, Serialize, Deserialize)]
pub struct PyModel {
    #[serde(flatten)]
    pub model: Arc<RwLock<ModelWrapper>>,
}

impl PyModel {
    pub(crate) fn new(model: Arc<RwLock<ModelWrapper>>) -> Self {
        PyModel { model }
    }

    pub(crate) fn get_as_subtype(&self) -> PyResult<PyObject> {
        let base = self.clone();
        let gil = Python::acquire_gil();
        let py = gil.python();
        Ok(match *self.model.as_ref().read().unwrap() {
            ModelWrapper::BPE(_) => Py::new(py, (PyBPE {}, base))?.into_py(py),
            ModelWrapper::WordPiece(_) => Py::new(py, (PyWordPiece {}, base))?.into_py(py),
            ModelWrapper::WordLevel(_) => Py::new(py, (PyWordLevel {}, base))?.into_py(py),
            ModelWrapper::Unigram(_) => Py::new(py, (PyUnigram {}, base))?.into_py(py),
        })
    }
}

impl Model for PyModel {
    type Trainer = PyTrainer;

    fn tokenize(&self, tokens: &str) -> tk::Result<Vec<Token>> {
        self.model.read().unwrap().tokenize(tokens)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.model.read().unwrap().token_to_id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.model.read().unwrap().id_to_token(id)
    }

    fn get_vocab(&self) -> HashMap<String, u32> {
        self.model.read().unwrap().get_vocab()
    }

    fn get_vocab_size(&self) -> usize {
        self.model.read().unwrap().get_vocab_size()
    }

    fn save(&self, folder: &Path, name: Option<&str>) -> tk::Result<Vec<PathBuf>> {
        self.model.read().unwrap().save(folder, name)
    }

    fn get_trainer(&self) -> Self::Trainer {
        self.model.read().unwrap().get_trainer().into()
    }
}

#[pymethods]
impl PyModel {
    #[new]
    fn __new__() -> PyResult<Self> {
        // Instantiate a default empty model. This doesn't really make sense, but we need
        // to be able to instantiate an empty model for pickle capabilities.
        Ok(PyModel {
            model: Arc::new(RwLock::new(BPE::default().into())),
        })
    }

    fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        let data = serde_json::to_string(&self.model).map_err(|e| {
            exceptions::PyException::new_err(format!(
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
                    exceptions::PyException::new_err(format!(
                        "Error while attempting to unpickle Model: {}",
                        e.to_string()
                    ))
                })?;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    /// Tokenize the given sequence
    #[text_signature = "(self, tokens)"]
    fn tokenize(&self, tokens: &str) -> PyResult<Vec<PyToken>> {
        Ok(ToPyResult(self.model.read().unwrap().tokenize(tokens))
            .into_py()?
            .into_iter()
            .map(|t| t.into())
            .collect())
    }

    /// Returns the id associated with the given token
    #[text_signature = "(self, tokens)"]
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.model.read().unwrap().token_to_id(token)
    }

    /// Returns the token associated with the given id
    #[text_signature = "(self, id)"]
    fn id_to_token(&self, id: u32) -> Option<String> {
        self.model.read().unwrap().id_to_token(id)
    }

    /// Save the current model
    ///
    /// Save the current model in the given folder, using the given name for the various
    /// files that will get created.
    /// Any file with the same name that already exist in this folder will be overwritten.
    #[text_signature = "(self, folder, name)"]
    fn save(&self, folder: &str, name: Option<&str>) -> PyResult<Vec<String>> {
        let saved: PyResult<Vec<_>> =
            ToPyResult(self.model.read().unwrap().save(Path::new(folder), name)).into();

        Ok(saved?
            .into_iter()
            .map(|path| path.to_string_lossy().into_owned())
            .collect())
    }

    fn get_trainer(&self) -> PyResult<PyObject> {
        PyTrainer::from(self.model.read().unwrap().get_trainer()).get_as_subtype()
    }
}

/// Instantiate a BPE Model from the given vocab and merges.
///
/// Args:
///    vocab: ('`optional`) Dict[str, int]:
///        A dictionnary of string keys and their ids {"am": 0,...}
///
///    merges: (`optional`) string:
///        A list of pairs of tokens [("a", "b"),...]
///
///    cache_capacity: (`optional`) int:
///        The number of words that the BPE cache can contain. The cache allows
///        to speed-up the process by keeping the result of the merge operations
///        for a number of words.
///
///    dropout: (`optional`) Optional[float] [0, 1]:
///        The BPE dropout to use. Must be an float between 0 and 1
///
///    unk_token: (`optional`) str:
///        The unknown token to be used by the model.
///
///    continuing_subword_prefix: (`optional`) str:
///        The prefix to attach to subword units that don't represent a beginning of word.
///
///    end_of_word_suffix: (`optional`) str:
///        The suffix to attach to subword units that represent an end of word.
///
///    fuse_unk: (`optional`) bool:
///        Multiple unk tokens get fused into only 1
#[pyclass(extends=PyModel, module = "tokenizers.models", name=BPE)]
#[text_signature = "(self, vocab=None, merges=None, cache_capacity=None, dropout=None, unk_token=None, continuing_subword_prefix=None, end_of_word_suffix=None, fuse_unk=None)"]
pub struct PyBPE {}

impl PyBPE {
    fn with_builder(mut builder: BpeBuilder, kwargs: Option<&PyDict>) -> PyResult<(Self, PyModel)> {
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
                    "fuse_unk" => builder = builder.fuse_unk(value.extract()?),
                    _ => println!("Ignored unknown kwarg option {}", key),
                };
            }
        }

        match builder.build() {
            Err(e) => Err(exceptions::PyException::new_err(format!(
                "Error while initializing BPE: {}",
                e
            ))),
            Ok(bpe) => Ok((PyBPE {}, PyModel::new(Arc::new(RwLock::new(bpe.into()))))),
        }
    }
}

#[derive(FromPyObject)]
enum PyVocab<'a> {
    Vocab(Vocab),
    Filename(&'a str),
}
#[derive(FromPyObject)]
enum PyMerges<'a> {
    Merges(Merges),
    Filename(&'a str),
}

#[pymethods]
impl PyBPE {
    #[new]
    #[args(kwargs = "**")]
    fn new(
        vocab: Option<PyVocab>,
        merges: Option<PyMerges>,
        kwargs: Option<&PyDict>,
    ) -> PyResult<(Self, PyModel)> {
        if (vocab.is_some() && merges.is_none()) || (vocab.is_none() && merges.is_some()) {
            return Err(exceptions::PyValueError::new_err(
                "`vocab` and `merges` must be both specified",
            ));
        }

        let mut builder = BPE::builder();
        if let (Some(vocab), Some(merges)) = (vocab, merges) {
            match (vocab, merges) {
                (PyVocab::Vocab(vocab), PyMerges::Merges(merges)) => {
                    builder = builder.vocab_and_merges(vocab, merges);
                }
                (PyVocab::Filename(vocab_filename), PyMerges::Filename(merges_filename)) => {
                    deprecation_warning(
                    "0.9.0",
                    "BPE.__init__ will not create from files anymore, try `BPE.from_file` instead",
                )?;
                    builder =
                        builder.files(vocab_filename.to_string(), merges_filename.to_string());
                }
                _ => {
                    return Err(exceptions::PyValueError::new_err(
                        "`vocab` and `merges` must be both be from memory or both filenames",
                    ));
                }
            }
        }

        PyBPE::with_builder(builder, kwargs)
    }

    /// Read a vocab_filename and merge_filename and stores result in memory
    #[staticmethod]
    #[text_signature = "(self, vocab_filename, merges_filename)"]
    fn read_file(vocab_filename: &str, merges_filename: &str) -> PyResult<(Vocab, Merges)> {
        BPE::read_file(vocab_filename, merges_filename).map_err(|e| {
            exceptions::PyValueError::new_err(format!(
                "Error while reading vocab&merges files: {}",
                e
            ))
        })
    }

    /// Convenient method to intialize a BPE from files
    /// Roughly equivalent to
    ///
    /// def from_file(vocab_filename, merges_filenames, **kwargs):
    ///     vocab, merges = BPE.read_file(vocab_filename, merges_filename)
    ///     return BPE(vocab, merges, **kwargs)
    #[staticmethod]
    #[args(kwargs = "**")]
    #[text_signature = "(vocab_filename, merge_filename, **kwargs)"]
    fn from_file(
        py: Python,
        vocab_filename: &str,
        merges_filename: &str,
        kwargs: Option<&PyDict>,
    ) -> PyResult<Py<Self>> {
        let (vocab, merges) = BPE::read_file(vocab_filename, merges_filename).map_err(|e| {
            exceptions::PyValueError::new_err(format!("Error while reading BPE files: {}", e))
        })?;
        Py::new(
            py,
            PyBPE::new(
                Some(PyVocab::Vocab(vocab)),
                Some(PyMerges::Merges(merges)),
                kwargs,
            )?,
        )
    }
}

/// WordPiece model
/// Instantiate a WordPiece Model from the given vocab file.
///
/// Args:
///     vocab: (`optional`) string:
///         A dictionnary of string keys and their ids {"am": 0,...}
///
///     unk_token: (`optional`) str:
///         The unknown token to be used by the model.
///
///     max_input_chars_per_word: (`optional`) int:
///         The maximum number of characters to authorize in a single word.
#[pyclass(extends=PyModel, module = "tokenizers.models", name=WordPiece)]
#[text_signature = "(self, vocab, unk_token, max_input_chars_per_word)"]
pub struct PyWordPiece {}

impl PyWordPiece {
    fn with_builder(
        mut builder: WordPieceBuilder,
        kwargs: Option<&PyDict>,
    ) -> PyResult<(Self, PyModel)> {
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
            Err(e) => Err(exceptions::PyException::new_err(format!(
                "Error while initializing WordPiece: {}",
                e
            ))),
            Ok(wordpiece) => Ok((
                PyWordPiece {},
                PyModel::new(Arc::new(RwLock::new(wordpiece.into()))),
            )),
        }
    }
}

#[pymethods]
impl PyWordPiece {
    #[new]
    #[args(kwargs = "**")]
    fn new(vocab: Option<PyVocab>, kwargs: Option<&PyDict>) -> PyResult<(Self, PyModel)> {
        let mut builder = WordPiece::builder();

        if let Some(vocab) = vocab {
            match vocab {
                PyVocab::Vocab(vocab) => {
                    builder = builder.vocab(vocab);
                }
                PyVocab::Filename(vocab_filename) => {
                    deprecation_warning(
                        "0.9.0",
                        "WordPiece.__init__ will not create from files anymore, try `WordPiece.from_file` instead",
                    )?;
                    builder = builder.files(vocab_filename.to_string());
                }
            }
        }

        PyWordPiece::with_builder(builder, kwargs)
    }

    /// Read a vocab_filename and stores result in memory
    #[staticmethod]
    #[text_signature = "(vocab_filename)"]
    fn read_file(vocab_filename: &str) -> PyResult<Vocab> {
        WordPiece::read_file(vocab_filename).map_err(|e| {
            exceptions::PyValueError::new_err(format!("Error while reading WordPiece file: {}", e))
        })
    }

    /// Convenient method to intialize a WordPiece from files
    /// Roughly equivalent to
    ///
    /// def from_file(vocab_filename, **kwargs):
    ///     vocab = WordPiece.read_file(vocab_filename)
    ///     return WordPiece(vocab, **kwargs)
    #[staticmethod]
    #[args(kwargs = "**")]
    #[text_signature = "(vocab_filename, merge_filename, **kwargs)"]
    fn from_file(py: Python, vocab_filename: &str, kwargs: Option<&PyDict>) -> PyResult<Py<Self>> {
        let vocab = WordPiece::read_file(vocab_filename).map_err(|e| {
            exceptions::PyValueError::new_err(format!("Error while reading WordPiece file: {}", e))
        })?;
        Py::new(py, PyWordPiece::new(Some(PyVocab::Vocab(vocab)), kwargs)?)
    }
}

/// Most simple tokenizer model based on mapping token from a vocab file to their corresponding id.
///
/// Instantiate a WordLevel Model from the given vocab file.
///
///     Args:
///         vocab: (`optional`) string:
///             A dictionnary of string keys and their ids {"am": 0,...}
///
///         unk_token: str:
///             The unknown token to be used by the model.
#[pyclass(extends=PyModel, module = "tokenizers.models", name=WordLevel)]
#[text_signature = "(self, vocab, unk_token)"]
pub struct PyWordLevel {}

impl PyWordLevel {
    fn get_unk(kwargs: Option<&PyDict>) -> PyResult<String> {
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
        Ok(unk_token)
    }
}

#[pymethods]
impl PyWordLevel {
    #[new]
    #[args(kwargs = "**")]
    fn new(vocab: Option<PyVocab>, kwargs: Option<&PyDict>) -> PyResult<(Self, PyModel)> {
        let unk_token = PyWordLevel::get_unk(kwargs)?;

        if let Some(vocab) = vocab {
            let model = match vocab {
                PyVocab::Vocab(vocab) => WordLevel::builder()
                    .vocab(vocab)
                    .unk_token(unk_token)
                    .build()
                    .expect("Can only fail when loading from files"),
                PyVocab::Filename(vocab_filename) => {
                    deprecation_warning(
                        "0.9.0",
                        "WordLevel.__init__ will not create from files anymore, \
                            try `WordLevel.from_file` instead",
                    )?;
                    WordLevel::from_file(vocab_filename, unk_token).map_err(|e| {
                        exceptions::PyException::new_err(format!(
                            "Error while loading WordLevel: {}",
                            e
                        ))
                    })?
                }
            };

            Ok((
                PyWordLevel {},
                PyModel::new(Arc::new(RwLock::new(model.into()))),
            ))
        } else {
            Ok((
                PyWordLevel {},
                PyModel::new(Arc::new(RwLock::new(WordLevel::default().into()))),
            ))
        }
    }

    #[staticmethod]
    fn read_file(vocab_filename: &str) -> PyResult<Vocab> {
        WordLevel::read_file(vocab_filename).map_err(|e| {
            exceptions::PyValueError::new_err(format!("Error while reading WordLevel file: {}", e))
        })
    }

    #[staticmethod]
    #[args(kwargs = "**")]
    fn from_file(py: Python, vocab_filename: &str, kwargs: Option<&PyDict>) -> PyResult<Py<Self>> {
        let vocab = WordLevel::read_file(vocab_filename).map_err(|e| {
            exceptions::PyValueError::new_err(format!("Error while reading WordLevel file: {}", e))
        })?;
        Py::new(py, PyWordLevel::new(Some(PyVocab::Vocab(vocab)), kwargs)?)
    }
}

/// UnigramEncoding model class
///
/// Instantiate a Unigram Model from the given model file.
///
/// Args:
///    vocab: ('`optional`) string:
///        A list of vocabulary items and their relative score [("am", -0.2442),...]
///
#[pyclass(extends=PyModel, module = "tokenizers.models", name=Unigram)]
#[text_signature = "(self, vocab)"]
pub struct PyUnigram {}

#[pymethods]
impl PyUnigram {
    #[new]
    fn new(vocab: Option<Vec<(String, f64)>>, unk_id: Option<usize>) -> PyResult<(Self, PyModel)> {
        match (vocab, unk_id) {
            (Some(vocab), unk_id) => {
                let model = Unigram::from(vocab, unk_id).map_err(|e| {
                    exceptions::PyException::new_err(format!("Error while loading Unigram: {}", e))
                })?;
                Ok((
                    PyUnigram {},
                    PyModel::new(Arc::new(RwLock::new(model.into()))),
                ))
            }
            (None, None) => Ok((
                PyUnigram {},
                PyModel::new(Arc::new(RwLock::new(Unigram::default().into()))),
            )),
            _ => Err(exceptions::PyValueError::new_err(
                "`vocab` and `unk_id` must be both specified",
            )),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::models::PyModel;
    use pyo3::prelude::*;
    use std::sync::{Arc, RwLock};
    use tk::models::bpe::BPE;
    use tk::models::ModelWrapper;

    #[test]
    fn get_subtype() {
        let py_model = PyModel::new(Arc::new(RwLock::new(BPE::default().into())));
        let py_bpe = py_model.get_as_subtype().unwrap();
        let gil = Python::acquire_gil();
        assert_eq!(
            "tokenizers.models.BPE",
            py_bpe.as_ref(gil.python()).get_type().name()
        );
    }

    #[test]
    fn serialize() {
        let rs_bpe = BPE::default();
        let rs_bpe_ser = serde_json::to_string(&rs_bpe).unwrap();
        let rs_wrapper: ModelWrapper = rs_bpe.into();
        let rs_wrapper_ser = serde_json::to_string(&rs_wrapper).unwrap();

        let py_model = PyModel::new(Arc::new(RwLock::new(rs_wrapper)));
        let py_ser = serde_json::to_string(&py_model).unwrap();
        assert_eq!(py_ser, rs_bpe_ser);
        assert_eq!(py_ser, rs_wrapper_ser);

        let py_model: PyModel = serde_json::from_str(&rs_bpe_ser).unwrap();
        match *py_model.model.as_ref().read().unwrap() {
            ModelWrapper::BPE(_) => (),
            _ => panic!("Expected Bert postprocessor."),
        }

        let py_model: PyModel = serde_json::from_str(&rs_wrapper_ser).unwrap();
        match *py_model.model.as_ref().read().unwrap() {
            ModelWrapper::BPE(_) => (),
            _ => panic!("Expected Bert postprocessor."),
        }
    }
}
