use std::collections::HashMap;
use std::sync::Arc;

use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::*;
use tk::models::TrainerWrapper;
use tk::Trainer;
use tokenizers as tk;

use crate::models::PyModel;
use crate::tokenizer::PyAddedToken;

/// Base class for all trainers
///
/// This class is not supposed to be instantiated directly. Instead, any implementation of a
/// Trainer will return an instance of this class when instantiated.
///
/// Args:
///     vocab_size: unsigned int:
///         The size of the final vocabulary, including all tokens and alphabet.
///
///     min_frequency: unsigned int:
///         The minimum frequency a pair should have in order to be merged.
///
///     show_progress: boolean:
///         Whether to show progress bars while training.
///
///     special_tokens: List[Union[str, AddedToken]]:
///         A list of special tokens the model should know of.
///
///     limit_alphabet: unsigned int:
///         The maximum different characters to keep in the alphabet.
///
///     initial_alphabet: List[str]:
///         A list of characters to include in the initial alphabet, even
///         if not seen in the training dataset.
///         If the strings contain more than one character, only the first one
///         is kept.
///
///     continuing_subword_prefix: Optional[str]:
///         A prefix to be used for every subword that is not a beginning-of-word.
///
///     end_of_word_suffix: Optional[str]:
///         A suffix to be used for every subword that is a end-of-word.
///
/// Returns:
///     Trainer
#[pyclass(name=Trainer)]
#[derive(Clone)]
#[text_signature = "(self, vocab_size=30000, min_frequency=0,show_progress=True, special_tokens=[],limit_alphabet=None, initial_alphabet = [], continuing_subword_prefix=None, end_of_word_suffix=None)"]
pub struct PyTrainer {
    pub trainer: Arc<TrainerWrapper>,
}

impl PyTrainer {
    pub(crate) fn new(trainer: Arc<TrainerWrapper>) -> Self {
        PyTrainer { trainer }
    }

    pub(crate) fn get_as_subtype(&self) -> PyResult<PyObject> {
        let base = self.clone();
        let gil = Python::acquire_gil();
        let py = gil.python();
        Ok(match self.trainer.as_ref() {
            TrainerWrapper::BpeTrainer(_) => Py::new(py, (PyBpeTrainer {}, base))?.into_py(py),
            TrainerWrapper::WordPieceTrainer(_) => {
                Py::new(py, (PyWordPieceTrainer {}, base))?.into_py(py)
            }
            TrainerWrapper::WordLevelTrainer(_) => {
                Py::new(py, (PyWordLevelTrainer {}, base))?.into_py(py)
            }
            TrainerWrapper::UnigramTrainer(_) => {
                Py::new(py, (PyUnigramTrainer {}, base))?.into_py(py)
            }
        })
    }
}

impl Trainer for PyTrainer {
    type Model = PyModel;

    fn should_show_progress(&self) -> bool {
        self.trainer.should_show_progress()
    }

    fn train(&self, words: HashMap<String, u32>) -> tk::Result<(PyModel, Vec<tk::AddedToken>)> {
        self.trainer.train(words).map(|(m, t)| {
            let m = PyModel { model: Arc::new(m) };
            (m, t)
        })
    }

    fn process_tokens(&self, words: &mut HashMap<String, u32>, tokens: Vec<String>) {
        self.trainer.process_tokens(words, tokens)
    }
}

impl<I> From<I> for PyTrainer
where
    I: Into<TrainerWrapper>,
{
    fn from(trainer: I) -> Self {
        PyTrainer {
            trainer: trainer.into().into(),
        }
    }
}

/// Capable of training a BPE model
#[pyclass(extends=PyTrainer, name=BpeTrainer)]
pub struct PyBpeTrainer {}
#[pymethods]
impl PyBpeTrainer {
    /// new(/ vocab_size, min_frequency)
    /// --
    ///
    /// Create a new BpeTrainer with the given configuration
    #[new]
    #[args(kwargs = "**")]
    pub fn new(kwargs: Option<&PyDict>) -> PyResult<(Self, PyTrainer)> {
        let mut builder = tk::models::bpe::BpeTrainer::builder();
        if let Some(kwargs) = kwargs {
            for (key, val) in kwargs {
                let key: &str = key.extract()?;
                match key {
                    "vocab_size" => builder = builder.vocab_size(val.extract()?),
                    "min_frequency" => builder = builder.min_frequency(val.extract()?),
                    "show_progress" => builder = builder.show_progress(val.extract()?),
                    "special_tokens" => {
                        builder = builder.special_tokens(
                            val.cast_as::<PyList>()?
                                .into_iter()
                                .map(|token| {
                                    if let Ok(content) = token.extract::<String>() {
                                        Ok(PyAddedToken::from(content, Some(true)).get_token())
                                    } else if let Ok(mut token) =
                                        token.extract::<PyRefMut<PyAddedToken>>()
                                    {
                                        token.is_special_token = true;
                                        Ok(token.get_token())
                                    } else {
                                        Err(exceptions::PyTypeError::new_err(
                                            "special_tokens must be a List[Union[str, AddedToken]]",
                                        ))
                                    }
                                })
                                .collect::<PyResult<Vec<_>>>()?,
                        );
                    }
                    "limit_alphabet" => builder = builder.limit_alphabet(val.extract()?),
                    "initial_alphabet" => {
                        let alphabet: Vec<String> = val.extract()?;
                        builder = builder.initial_alphabet(
                            alphabet
                                .into_iter()
                                .map(|s| s.chars().next())
                                .filter(|c| c.is_some())
                                .map(|c| c.unwrap())
                                .collect(),
                        );
                    }
                    "continuing_subword_prefix" => {
                        builder = builder.continuing_subword_prefix(val.extract()?)
                    }
                    "end_of_word_suffix" => builder = builder.end_of_word_suffix(val.extract()?),
                    _ => println!("Ignored unknown kwargs option {}", key),
                };
            }
        }
        Ok((
            PyBpeTrainer {},
            PyTrainer::new(Arc::new(builder.build().into())),
        ))
    }
}

/// Capable of training a WordPiece model
/// Args:
///     vocab_size: unsigned int:
///         The size of the final vocabulary, including all tokens and alphabet.
///
///     min_frequency: unsigned int:
///         The minimum frequency a pair should have in order to be merged.
///
///     show_progress: boolean:
///         Whether to show progress bars while training.
///
///     special_tokens: List[Union[str, AddedToken]]:
///         A list of special tokens the model should know of.
///
///     limit_alphabet: unsigned int:
///         The maximum different characters to keep in the alphabet.
///
///     initial_alphabet: List[str]:
///         A list of characters to include in the initial alphabet, even
///         if not seen in the training dataset.
///         If the strings contain more than one character, only the first one
///         is kept.
///
///     continuing_subword_prefix: Optional[str]:
///         A prefix to be used for every subword that is not a beginning-of-word.
///
///     end_of_word_suffix: Optional[str]:
///         A suffix to be used for every subword that is a end-of-word.
///
/// Returns:
///     Trainer
#[pyclass(extends=PyTrainer, name=WordPieceTrainer)]
#[text_signature = "(self, vocab_size=30000, min_frequency=0, show_progress=True, special_tokens=[], limit_alphabet=None, initial_alphabet= [],continuing_subword_prefix=\"##\", end_of_word_suffix=None)"]
pub struct PyWordPieceTrainer {}
#[pymethods]
impl PyWordPieceTrainer {
    /// new(/ vocab_size, min_frequency)
    /// --
    ///
    /// Create a new BpeTrainer with the given configuration
    #[new]
    #[args(kwargs = "**")]
    pub fn new(kwargs: Option<&PyDict>) -> PyResult<(Self, PyTrainer)> {
        let mut builder = tk::models::wordpiece::WordPieceTrainer::builder();
        if let Some(kwargs) = kwargs {
            for (key, val) in kwargs {
                let key: &str = key.extract()?;
                match key {
                    "vocab_size" => builder = builder.vocab_size(val.extract()?),
                    "min_frequency" => builder = builder.min_frequency(val.extract()?),
                    "show_progress" => builder = builder.show_progress(val.extract()?),
                    "special_tokens" => {
                        builder = builder.special_tokens(
                            val.cast_as::<PyList>()?
                                .into_iter()
                                .map(|token| {
                                    if let Ok(content) = token.extract::<String>() {
                                        Ok(PyAddedToken::from(content, Some(true)).get_token())
                                    } else if let Ok(mut token) =
                                        token.extract::<PyRefMut<PyAddedToken>>()
                                    {
                                        token.is_special_token = true;
                                        Ok(token.get_token())
                                    } else {
                                        Err(exceptions::PyTypeError::new_err(
                                            "special_tokens must be a List[Union[str, AddedToken]]",
                                        ))
                                    }
                                })
                                .collect::<PyResult<Vec<_>>>()?,
                        );
                    }
                    "limit_alphabet" => builder = builder.limit_alphabet(val.extract()?),
                    "initial_alphabet" => {
                        let alphabet: Vec<String> = val.extract()?;
                        builder = builder.initial_alphabet(
                            alphabet
                                .into_iter()
                                .map(|s| s.chars().next())
                                .filter(|c| c.is_some())
                                .map(|c| c.unwrap())
                                .collect(),
                        );
                    }
                    "continuing_subword_prefix" => {
                        builder = builder.continuing_subword_prefix(val.extract()?)
                    }
                    "end_of_word_suffix" => builder = builder.end_of_word_suffix(val.extract()?),
                    _ => println!("Ignored unknown kwargs option {}", key),
                };
            }
        }

        Ok((
            PyWordPieceTrainer {},
            PyTrainer::new(Arc::new(builder.build().into())),
        ))
    }
}

/// Capable of training a WorldLevel model
///
/// Args:
///     vocab_size: unsigned int:
///         The size of the final vocabulary, including all tokens and alphabet.
///
///     min_frequency: unsigned int:
///         The minimum frequency a pair should have in order to be merged.
///
///     show_progress: boolean:
///         Whether to show progress bars while training.
///
///     special_tokens: List[Union[str, AddedToken]]:
///         A list of special tokens the model should know of.
///
/// Returns:
///     Trainer
#[pyclass(extends=PyTrainer, name=WordLevelTrainer)]
pub struct PyWordLevelTrainer {}
#[pymethods]
impl PyWordLevelTrainer {
    /// Create a new WordLevelTrainer with the given configuration
    #[new]
    #[args(kwargs = "**")]
    pub fn new(kwargs: Option<&PyDict>) -> PyResult<(Self, PyTrainer)> {
        let mut trainer = tk::models::wordlevel::WordLevelTrainer::default();

        if let Some(kwargs) = kwargs {
            for (key, val) in kwargs {
                let key: &str = key.extract()?;
                match key {
                    "vocab_size" => trainer.vocab_size = val.extract()?,
                    "min_frequency" => trainer.min_frequency = val.extract()?,
                    "show_progress" => trainer.show_progress = val.extract()?,
                    "special_tokens" => {
                        trainer.special_tokens = val
                            .cast_as::<PyList>()?
                            .into_iter()
                            .map(|token| {
                                if let Ok(content) = token.extract::<String>() {
                                    Ok(PyAddedToken::from(content, Some(true)).get_token())
                                } else if let Ok(mut token) =
                                    token.extract::<PyRefMut<PyAddedToken>>()
                                {
                                    token.is_special_token = true;
                                    Ok(token.get_token())
                                } else {
                                    Err(exceptions::PyTypeError::new_err(
                                        "special_tokens must be a List[Union[str, AddedToken]]",
                                    ))
                                }
                            })
                            .collect::<PyResult<Vec<_>>>()?
                    }
                    _ => println!("Ignored unknown kwargs option {}", key),
                }
            }
        }

        Ok((
            PyWordLevelTrainer {},
            PyTrainer::new(Arc::new(trainer.into())),
        ))
    }
}

/// Capable of training a Unigram model
///
/// Args:
///     vocab_size: unsigned int:
///         The size of the final vocabulary, including all tokens and alphabet.
///
///     show_progress: boolean:
///         Whether to show progress bars while training.
///
///     special_tokens: List[Union[str, AddedToken]]:
///         A list of special tokens the model should know of.
///
///     initial_alphabet: List[str]:
///         A list of characters to include in the initial alphabet, even
///         if not seen in the training dataset.
///         If the strings contain more than one character, only the first one
///         is kept.
///
/// Returns:
///     Trainer
#[pyclass(extends=PyTrainer, name=UnigramTrainer)]
#[text_signature = "(self, vocab_size=8000, show_progress=True, special_tokens= [])"]
pub struct PyUnigramTrainer {}
#[pymethods]
impl PyUnigramTrainer {
    /// Create a new UnigramTrainer with the given configuration
    #[new]
    #[args(kwargs = "**")]
    pub fn new(kwargs: Option<&PyDict>) -> PyResult<(Self, PyTrainer)> {
        let mut builder = tk::models::unigram::UnigramTrainer::builder();
        if let Some(kwargs) = kwargs {
            for (key, val) in kwargs {
                let key: &str = key.extract()?;
                match key {
                    "vocab_size" => builder.vocab_size(val.extract()?),
                    "show_progress" => builder.show_progress(val.extract()?),
                    "n_sub_iterations" => builder.n_sub_iterations(val.extract()?),
                    "shrinking_factor" => builder.shrinking_factor(val.extract()?),
                    "unk_token" => builder.unk_token(val.extract()?),
                    "max_piece_length" => builder.max_piece_length(val.extract()?),
                    "seed_size" => builder.seed_size(val.extract()?),
                    "initial_alphabet" => {
                        let alphabet: Vec<String> = val.extract()?;
                        builder.initial_alphabet(
                            alphabet
                                .into_iter()
                                .map(|s| s.chars().next())
                                .filter(|c| c.is_some())
                                .map(|c| c.unwrap())
                                .collect(),
                        )
                    }
                    "special_tokens" => builder.special_tokens(
                        val.cast_as::<PyList>()?
                            .into_iter()
                            .map(|token| {
                                if let Ok(content) = token.extract::<String>() {
                                    Ok(PyAddedToken::from(content, Some(true)).get_token())
                                } else if let Ok(mut token) =
                                    token.extract::<PyRefMut<PyAddedToken>>()
                                {
                                    token.is_special_token = true;
                                    Ok(token.get_token())
                                } else {
                                    Err(exceptions::PyTypeError::new_err(
                                        "special_tokens must be a List[Union[str, AddedToken]]",
                                    ))
                                }
                            })
                            .collect::<PyResult<Vec<_>>>()?,
                    ),
                    _ => {
                        println!("Ignored unknown kwargs option {}", key);
                        &mut builder
                    }
                };
            }
        }

        let trainer: tokenizers::models::unigram::UnigramTrainer =
            builder.build().map_err(|e| {
                exceptions::PyException::new_err(format!("Cannot build UnigramTrainer: {}", e))
            })?;
        Ok((
            PyUnigramTrainer {},
            PyTrainer::new(Arc::new(trainer.into())),
        ))
    }
}
