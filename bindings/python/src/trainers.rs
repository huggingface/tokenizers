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

#[pyclass(name=Trainer)]
pub struct PyTrainer {
    pub trainer: TrainerWrapper,
}

impl PyTrainer {
    pub fn new(trainer: TrainerWrapper) -> Self {
        PyTrainer { trainer }
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
        Ok((PyBpeTrainer {}, PyTrainer::new(builder.build().into())))
    }
}

#[pyclass(extends=PyTrainer, name=WordPieceTrainer)]
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
            PyTrainer::new(builder.build().into()),
        ))
    }
}

#[pyclass(extends=PyTrainer, name=UnigramTrainer)]
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
        Ok((PyUnigramTrainer {}, PyTrainer::new(trainer.into())))
    }
}
