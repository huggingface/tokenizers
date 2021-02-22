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

/// Base class for all models
///
/// The model represents the actual tokenization algorithm. This is the part that
/// will contain and manage the learned vocabulary.
///
/// This class cannot be constructed directly. Please use one of the concrete models.
#[pyclass(module = "tokenizers.models", name=Model)]
#[derive(Clone, Serialize, Deserialize)]
pub struct PyModel {
    #[serde(flatten)]
    pub model: Arc<RwLock<ModelWrapper>>,
}

impl PyModel {
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

impl<I> From<I> for PyModel
where
    I: Into<ModelWrapper>,
{
    fn from(model: I) -> Self {
        Self {
            model: Arc::new(RwLock::new(model.into())),
        }
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

    /// Tokenize a sequence
    ///
    /// Args:
    ///     sequence (:obj:`str`):
    ///         A sequence to tokenize
    ///
    /// Returns:
    ///     A :obj:`List` of :class:`~tokenizers.Token`: The generated tokens
    #[text_signature = "(self, sequence)"]
    fn tokenize(&self, sequence: &str) -> PyResult<Vec<PyToken>> {
        Ok(ToPyResult(self.model.read().unwrap().tokenize(sequence))
            .into_py()?
            .into_iter()
            .map(|t| t.into())
            .collect())
    }

    /// Get the ID associated to a token
    ///
    /// Args:
    ///     token (:obj:`str`):
    ///         A token to convert to an ID
    ///
    /// Returns:
    ///     :obj:`int`: The ID associated to the token
    #[text_signature = "(self, tokens)"]
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.model.read().unwrap().token_to_id(token)
    }

    /// Get the token associated to an ID
    ///
    /// Args:
    ///     id (:obj:`int`):
    ///         An ID to convert to a token
    ///
    /// Returns:
    ///     :obj:`str`: The token associated to the ID
    #[text_signature = "(self, id)"]
    fn id_to_token(&self, id: u32) -> Option<String> {
        self.model.read().unwrap().id_to_token(id)
    }

    /// Save the current model
    ///
    /// Save the current model in the given folder, using the given prefix for the various
    /// files that will get created.
    /// Any file with the same name that already exists in this folder will be overwritten.
    ///
    /// Args:
    ///     folder (:obj:`str`):
    ///         The path to the target folder in which to save the various files
    ///
    ///     prefix (:obj:`str`, `optional`):
    ///         An optional prefix, used to prefix each file name
    ///
    /// Returns:
    ///     :obj:`List[str]`: The list of saved files
    #[text_signature = "(self, folder, prefix)"]
    fn save(&self, folder: &str, prefix: Option<&str>) -> PyResult<Vec<String>> {
        let saved: PyResult<Vec<_>> =
            ToPyResult(self.model.read().unwrap().save(Path::new(folder), prefix)).into();

        Ok(saved?
            .into_iter()
            .map(|path| path.to_string_lossy().into_owned())
            .collect())
    }

    /// Get the associated :class:`~tokenizers.trainers.Trainer`
    ///
    /// Retrieve the :class:`~tokenizers.trainers.Trainer` associated to this
    /// :class:`~tokenizers.models.Model`.
    ///
    /// Returns:
    ///     :class:`~tokenizers.trainers.Trainer`: The Trainer used to train this model
    fn get_trainer(&self) -> PyResult<PyObject> {
        PyTrainer::from(self.model.read().unwrap().get_trainer()).get_as_subtype()
    }
}

/// An implementation of the BPE (Byte-Pair Encoding) algorithm
///
/// Args:
///     vocab (:obj:`Dict[str, int]`, `optional`):
///         A dictionnary of string keys and their ids :obj:`{"am": 0,...}`
///
///     merges (:obj:`List[Tuple[str, str]]`, `optional`):
///         A list of pairs of tokens (:obj:`Tuple[str, str]`) :obj:`[("a", "b"),...]`
///
///     cache_capacity (:obj:`int`, `optional`):
///         The number of words that the BPE cache can contain. The cache allows
///         to speed-up the process by keeping the result of the merge operations
///         for a number of words.
///
///     dropout (:obj:`float`, `optional`):
///         A float between 0 and 1 that represents the BPE dropout to use.
///
///     unk_token (:obj:`str`, `optional`):
///         The unknown token to be used by the model.
///
///     continuing_subword_prefix (:obj:`str`, `optional`):
///         The prefix to attach to subword units that don't represent a beginning of word.
///
///     end_of_word_suffix (:obj:`str`, `optional`):
///         The suffix to attach to subword units that represent an end of word.
///
///     fuse_unk (:obj:`bool`, `optional`):
///         Whether to fuse any subsequent unknown tokens into a single one
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
            Ok(bpe) => Ok((PyBPE {}, bpe.into())),
        }
    }
}

macro_rules! getter {
    ($self: ident, $variant: ident, $($name: tt)+) => {{
        let super_ = $self.as_ref();
        let model = super_.model.read().unwrap();
        if let ModelWrapper::$variant(ref mo) = *model {
            mo.$($name)+
        } else {
            unreachable!()
        }
    }};
}

macro_rules! setter {
    ($self: ident, $variant: ident, $name: ident, $value: expr) => {{
        let super_ = $self.as_ref();
        let mut model = super_.model.write().unwrap();
        if let ModelWrapper::$variant(ref mut mo) = *model {
            mo.$name = $value;
        }
    }};
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
    #[getter]
    fn get_dropout(self_: PyRef<Self>) -> Option<f32> {
        getter!(self_, BPE, dropout)
    }

    #[setter]
    fn set_dropout(self_: PyRef<Self>, dropout: Option<f32>) {
        setter!(self_, BPE, dropout, dropout);
    }

    #[getter]
    fn get_unk_token(self_: PyRef<Self>) -> Option<String> {
        getter!(self_, BPE, unk_token.clone())
    }

    #[setter]
    fn set_unk_token(self_: PyRef<Self>, unk_token: Option<String>) {
        setter!(self_, BPE, unk_token, unk_token);
    }

    #[getter]
    fn get_continuing_subword_prefix(self_: PyRef<Self>) -> Option<String> {
        getter!(self_, BPE, continuing_subword_prefix.clone())
    }

    #[setter]
    fn set_continuing_subword_prefix(
        self_: PyRef<Self>,
        continuing_subword_prefix: Option<String>,
    ) {
        setter!(
            self_,
            BPE,
            continuing_subword_prefix,
            continuing_subword_prefix
        );
    }

    #[getter]
    fn get_end_of_word_suffix(self_: PyRef<Self>) -> Option<String> {
        getter!(self_, BPE, end_of_word_suffix.clone())
    }

    #[setter]
    fn set_end_of_word_suffix(self_: PyRef<Self>, end_of_word_suffix: Option<String>) {
        setter!(self_, BPE, end_of_word_suffix, end_of_word_suffix);
    }

    #[getter]
    fn get_fuse_unk(self_: PyRef<Self>) -> bool {
        getter!(self_, BPE, fuse_unk)
    }

    #[setter]
    fn set_fuse_unk(self_: PyRef<Self>, fuse_unk: bool) {
        setter!(self_, BPE, fuse_unk, fuse_unk);
    }

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

    /// Read a :obj:`vocab.json` and a :obj:`merges.txt` files
    ///
    /// This method provides a way to read and parse the content of these files,
    /// returning the relevant data structures. If you want to instantiate some BPE models
    /// from memory, this method gives you the expected input from the standard files.
    ///
    /// Args:
    ///     vocab (:obj:`str`):
    ///         The path to a :obj:`vocab.json` file
    ///
    ///     merges (:obj:`str`):
    ///         The path to a :obj:`merges.txt` file
    ///
    /// Returns:
    ///     A :obj:`Tuple` with the vocab and the merges:
    ///         The vocabulary and merges loaded into memory
    #[staticmethod]
    #[text_signature = "(self, vocab, merges)"]
    fn read_file(vocab: &str, merges: &str) -> PyResult<(Vocab, Merges)> {
        BPE::read_file(vocab, merges).map_err(|e| {
            exceptions::PyException::new_err(format!(
                "Error while reading vocab & merges files: {}",
                e
            ))
        })
    }

    /// Instantiate a BPE model from the given files.
    ///
    /// This method is roughly equivalent to doing::
    ///
    ///    vocab, merges = BPE.read_file(vocab_filename, merges_filename)
    ///    bpe = BPE(vocab, merges)
    ///
    /// If you don't need to keep the :obj:`vocab, merges` values lying around,
    /// this method is more optimized than manually calling
    /// :meth:`~tokenizers.models.BPE.read_file` to initialize a :class:`~tokenizers.models.BPE`
    ///
    /// Args:
    ///     vocab (:obj:`str`):
    ///         The path to a :obj:`vocab.json` file
    ///
    ///     merges (:obj:`str`):
    ///         The path to a :obj:`merges.txt` file
    ///
    /// Returns:
    ///     :class:`~tokenizers.models.BPE`: An instance of BPE loaded from these files
    #[classmethod]
    #[args(kwargs = "**")]
    #[text_signature = "(cls, vocab, merge, **kwargs)"]
    fn from_file(
        _cls: &PyType,
        py: Python,
        vocab: &str,
        merges: &str,
        kwargs: Option<&PyDict>,
    ) -> PyResult<Py<Self>> {
        let (vocab, merges) = BPE::read_file(vocab, merges).map_err(|e| {
            exceptions::PyException::new_err(format!("Error while reading BPE files: {}", e))
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

/// An implementation of the WordPiece algorithm
///
/// Args:
///     vocab (:obj:`Dict[str, int]`, `optional`):
///         A dictionnary of string keys and their ids :obj:`{"am": 0,...}`
///
///     unk_token (:obj:`str`, `optional`):
///         The unknown token to be used by the model.
///
///     max_input_chars_per_word (:obj:`int`, `optional`):
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
            Ok(wordpiece) => Ok((PyWordPiece {}, wordpiece.into())),
        }
    }
}

#[pymethods]
impl PyWordPiece {
    #[getter]
    fn get_unk_token(self_: PyRef<Self>) -> String {
        getter!(self_, WordPiece, unk_token.clone())
    }

    #[setter]
    fn set_unk_token(self_: PyRef<Self>, unk_token: String) {
        setter!(self_, WordPiece, unk_token, unk_token);
    }

    #[getter]
    fn get_continuing_subword_prefix(self_: PyRef<Self>) -> String {
        getter!(self_, WordPiece, continuing_subword_prefix.clone())
    }

    #[setter]
    fn set_continuing_subword_prefix(self_: PyRef<Self>, continuing_subword_prefix: String) {
        setter!(
            self_,
            WordPiece,
            continuing_subword_prefix,
            continuing_subword_prefix
        );
    }

    #[getter]
    fn get_max_input_chars_per_word(self_: PyRef<Self>) -> usize {
        getter!(self_, WordPiece, max_input_chars_per_word)
    }

    #[setter]
    fn set_max_input_chars_per_word(self_: PyRef<Self>, max: usize) {
        setter!(self_, WordPiece, max_input_chars_per_word, max);
    }

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

    /// Read a :obj:`vocab.txt` file
    ///
    /// This method provides a way to read and parse the content of a standard `vocab.txt`
    /// file as used by the WordPiece Model, returning the relevant data structures. If you
    /// want to instantiate some WordPiece models from memory, this method gives you the
    /// expected input from the standard files.
    ///
    /// Args:
    ///     vocab (:obj:`str`):
    ///         The path to a :obj:`vocab.txt` file
    ///
    /// Returns:
    ///     :obj:`Dict[str, int]`: The vocabulary as a :obj:`dict`
    #[staticmethod]
    #[text_signature = "(vocab)"]
    fn read_file(vocab: &str) -> PyResult<Vocab> {
        WordPiece::read_file(vocab).map_err(|e| {
            exceptions::PyException::new_err(format!("Error while reading WordPiece file: {}", e))
        })
    }

    /// Instantiate a WordPiece model from the given file
    ///
    /// This method is roughly equivalent to doing::
    ///
    ///     vocab = WordPiece.read_file(vocab_filename)
    ///     wordpiece = WordPiece(vocab)
    ///
    /// If you don't need to keep the :obj:`vocab` values lying around, this method is
    /// more optimized than manually calling :meth:`~tokenizers.models.WordPiece.read_file` to
    /// initialize a :class:`~tokenizers.models.WordPiece`
    ///
    /// Args:
    ///     vocab (:obj:`str`):
    ///         The path to a :obj:`vocab.txt` file
    ///
    /// Returns:
    ///     :class:`~tokenizers.models.WordPiece`: And instance of WordPiece loaded from file
    #[classmethod]
    #[args(kwargs = "**")]
    #[text_signature = "(vocab, **kwargs)"]
    fn from_file(
        _cls: &PyType,
        py: Python,
        vocab: &str,
        kwargs: Option<&PyDict>,
    ) -> PyResult<Py<Self>> {
        let vocab = WordPiece::read_file(vocab).map_err(|e| {
            exceptions::PyException::new_err(format!("Error while reading WordPiece file: {}", e))
        })?;
        Py::new(py, PyWordPiece::new(Some(PyVocab::Vocab(vocab)), kwargs)?)
    }
}

/// An implementation of the WordLevel algorithm
///
/// Most simple tokenizer model based on mapping tokens to their corresponding id.
///
/// Args:
///     vocab (:obj:`str`, `optional`):
///         A dictionnary of string keys and their ids :obj:`{"am": 0,...}`
///
///     unk_token (:obj:`str`, `optional`):
///         The unknown token to be used by the model.
#[pyclass(extends=PyModel, module = "tokenizers.models", name=WordLevel)]
#[text_signature = "(self, vocab, unk_token)"]
pub struct PyWordLevel {}

#[pymethods]
impl PyWordLevel {
    #[getter]
    fn get_unk_token(self_: PyRef<Self>) -> String {
        getter!(self_, WordLevel, unk_token.clone())
    }

    #[setter]
    fn set_unk_token(self_: PyRef<Self>, unk_token: String) {
        setter!(self_, WordLevel, unk_token, unk_token);
    }

    #[new]
    #[args(unk_token = "None")]
    fn new(vocab: Option<PyVocab>, unk_token: Option<String>) -> PyResult<(Self, PyModel)> {
        let mut builder = WordLevel::builder();

        if let Some(vocab) = vocab {
            match vocab {
                PyVocab::Vocab(vocab) => {
                    builder = builder.vocab(vocab);
                }
                PyVocab::Filename(vocab_filename) => {
                    deprecation_warning(
                        "0.9.0",
                        "WordLevel.__init__ will not create from files anymore, \
                            try `WordLevel.from_file` instead",
                    )?;
                    builder = builder.files(vocab_filename.to_string());
                }
            };
        }
        if let Some(unk_token) = unk_token {
            builder = builder.unk_token(unk_token);
        }

        Ok((
            PyWordLevel {},
            builder
                .build()
                .map_err(|e| exceptions::PyException::new_err(e.to_string()))?
                .into(),
        ))
    }

    /// Read a :obj:`vocab.json`
    ///
    /// This method provides a way to read and parse the content of a vocabulary file,
    /// returning the relevant data structures. If you want to instantiate some WordLevel models
    /// from memory, this method gives you the expected input from the standard files.
    ///
    /// Args:
    ///     vocab (:obj:`str`):
    ///         The path to a :obj:`vocab.json` file
    ///
    /// Returns:
    ///     :obj:`Dict[str, int]`: The vocabulary as a :obj:`dict`
    #[staticmethod]
    #[text_signature = "(vocab)"]
    fn read_file(vocab: &str) -> PyResult<Vocab> {
        WordLevel::read_file(vocab).map_err(|e| {
            exceptions::PyException::new_err(format!("Error while reading WordLevel file: {}", e))
        })
    }

    /// Instantiate a WordLevel model from the given file
    ///
    /// This method is roughly equivalent to doing::
    ///
    ///     vocab = WordLevel.read_file(vocab_filename)
    ///     wordlevel = WordLevel(vocab)
    ///
    /// If you don't need to keep the :obj:`vocab` values lying around, this method is
    /// more optimized than manually calling :meth:`~tokenizers.models.WordLevel.read_file` to
    /// initialize a :class:`~tokenizers.models.WordLevel`
    ///
    /// Args:
    ///     vocab (:obj:`str`):
    ///         The path to a :obj:`vocab.json` file
    ///
    /// Returns:
    ///     :class:`~tokenizers.models.WordLevel`: And instance of WordLevel loaded from file
    #[classmethod]
    #[args(unk_token = "None")]
    fn from_file(
        _cls: &PyType,
        py: Python,
        vocab: &str,
        unk_token: Option<String>,
    ) -> PyResult<Py<Self>> {
        let vocab = WordLevel::read_file(vocab).map_err(|e| {
            exceptions::PyException::new_err(format!("Error while reading WordLevel file: {}", e))
        })?;
        Py::new(
            py,
            PyWordLevel::new(Some(PyVocab::Vocab(vocab)), unk_token)?,
        )
    }
}

/// An implementation of the Unigram algorithm
///
/// Args:
///     vocab (:obj:`List[Tuple[str, float]]`, `optional`):
///         A list of vocabulary items and their relative score [("am", -0.2442),...]
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
                Ok((PyUnigram {}, model.into()))
            }
            (None, None) => Ok((PyUnigram {}, Unigram::default().into())),
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
    use tk::models::bpe::BPE;
    use tk::models::ModelWrapper;

    #[test]
    fn get_subtype() {
        let py_model = PyModel::from(BPE::default());
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

        let py_model = PyModel::from(rs_wrapper);
        let py_ser = serde_json::to_string(&py_model).unwrap();
        assert_eq!(py_ser, rs_bpe_ser);
        assert_eq!(py_ser, rs_wrapper_ser);

        let py_model: PyModel = serde_json::from_str(&rs_bpe_ser).unwrap();
        match *py_model.model.as_ref().read().unwrap() {
            ModelWrapper::BPE(_) => (),
            _ => panic!("Expected Bert postprocessor."),
        };

        let py_model: PyModel = serde_json::from_str(&rs_wrapper_ser).unwrap();
        match *py_model.model.as_ref().read().unwrap() {
            ModelWrapper::BPE(_) => (),
            _ => panic!("Expected Bert postprocessor."),
        };
    }
}
