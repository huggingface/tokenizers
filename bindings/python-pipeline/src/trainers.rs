use pyo3::prelude::*;
use tk_train::trainers::{
    BpeTrainer, TrainerWrapper, UnigramTrainer, WordLevelTrainer, WordPieceTrainer,
};

use crate::added_token::{TokenInput, parse_tokens};
use crate::error::to_pyerr;

/// Base class for all trainers. A trainer is a plain configuration value:
/// `Tokenizer.train*` copies it, no state is shared or written back.
#[pyclass(
    frozen,
    subclass,
    name = "Trainer",
    module = "tokenizers_pipeline.trainers"
)]
pub struct PyTrainer {
    pub inner: TrainerWrapper,
}

#[pymethods]
impl PyTrainer {
    fn __repr__(&self) -> String {
        crate::component_repr(&self.inner)
    }
}

#[pyclass(frozen, extends = PyTrainer, name = "BpeTrainer", module = "tokenizers_pipeline.trainers")]
pub struct PyBpeTrainer;

#[pymethods]
impl PyBpeTrainer {
    #[new]
    #[pyo3(signature = (*, vocab_size = 30000, min_frequency = 0, special_tokens = vec![], limit_alphabet = None, initial_alphabet = vec![], continuing_subword_prefix = None, end_of_word_suffix = None, max_token_length = None, show_progress = true))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        vocab_size: usize,
        min_frequency: u64,
        special_tokens: Vec<TokenInput>,
        limit_alphabet: Option<usize>,
        initial_alphabet: Vec<char>,
        continuing_subword_prefix: Option<String>,
        end_of_word_suffix: Option<String>,
        max_token_length: Option<usize>,
        show_progress: bool,
    ) -> PyResult<PyClassInitializer<Self>> {
        let mut builder = BpeTrainer::builder()
            .vocab_size(vocab_size)
            .min_frequency(min_frequency)
            .special_tokens(parse_tokens(special_tokens, true))
            .initial_alphabet(initial_alphabet.into_iter().collect())
            .show_progress(show_progress);
        if let Some(limit) = limit_alphabet {
            builder = builder.limit_alphabet(limit);
        }
        if let Some(prefix) = continuing_subword_prefix {
            builder = builder.continuing_subword_prefix(prefix);
        }
        if let Some(suffix) = end_of_word_suffix {
            builder = builder.end_of_word_suffix(suffix);
        }
        if let Some(max) = max_token_length {
            builder = builder.max_token_length(Some(max));
        }
        Ok(PyClassInitializer::from(PyTrainer {
            inner: builder.build().into(),
        })
        .add_subclass(PyBpeTrainer))
    }
}

#[pyclass(frozen, extends = PyTrainer, name = "WordPieceTrainer", module = "tokenizers_pipeline.trainers")]
pub struct PyWordPieceTrainer;

#[pymethods]
impl PyWordPieceTrainer {
    #[new]
    #[pyo3(signature = (*, vocab_size = 30000, min_frequency = 0, special_tokens = vec![], limit_alphabet = None, initial_alphabet = vec![], continuing_subword_prefix = String::from("##"), end_of_word_suffix = None, show_progress = true))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        vocab_size: usize,
        min_frequency: u64,
        special_tokens: Vec<TokenInput>,
        limit_alphabet: Option<usize>,
        initial_alphabet: Vec<char>,
        continuing_subword_prefix: String,
        end_of_word_suffix: Option<String>,
        show_progress: bool,
    ) -> PyResult<PyClassInitializer<Self>> {
        let mut builder = WordPieceTrainer::builder()
            .vocab_size(vocab_size)
            .min_frequency(min_frequency)
            .special_tokens(parse_tokens(special_tokens, true))
            .initial_alphabet(initial_alphabet.into_iter().collect())
            .continuing_subword_prefix(continuing_subword_prefix)
            .show_progress(show_progress);
        if let Some(limit) = limit_alphabet {
            builder = builder.limit_alphabet(limit);
        }
        if let Some(suffix) = end_of_word_suffix {
            builder = builder.end_of_word_suffix(suffix);
        }
        Ok(PyClassInitializer::from(PyTrainer {
            inner: builder.build().into(),
        })
        .add_subclass(PyWordPieceTrainer))
    }
}

#[pyclass(frozen, extends = PyTrainer, name = "UnigramTrainer", module = "tokenizers_pipeline.trainers")]
pub struct PyUnigramTrainer;

#[pymethods]
impl PyUnigramTrainer {
    #[new]
    #[pyo3(signature = (*, vocab_size = 8000, special_tokens = vec![], initial_alphabet = vec![], unk_token = None, shrinking_factor = 0.75, max_piece_length = 16, n_sub_iterations = 2, show_progress = true))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        vocab_size: u32,
        special_tokens: Vec<TokenInput>,
        initial_alphabet: Vec<char>,
        unk_token: Option<String>,
        shrinking_factor: f64,
        max_piece_length: usize,
        n_sub_iterations: u32,
        show_progress: bool,
    ) -> PyResult<PyClassInitializer<Self>> {
        let trainer = UnigramTrainer::builder()
            .vocab_size(vocab_size)
            .special_tokens(parse_tokens(special_tokens, true))
            .initial_alphabet(initial_alphabet.into_iter().collect())
            .unk_token(unk_token)
            .shrinking_factor(shrinking_factor)
            .max_piece_length(max_piece_length)
            .n_sub_iterations(n_sub_iterations)
            .show_progress(show_progress)
            .build()
            .map_err(|e| to_pyerr(e.to_string().into()))?;
        Ok(PyClassInitializer::from(PyTrainer {
            inner: trainer.into(),
        })
        .add_subclass(PyUnigramTrainer))
    }
}

#[pyclass(frozen, extends = PyTrainer, name = "WordLevelTrainer", module = "tokenizers_pipeline.trainers")]
pub struct PyWordLevelTrainer;

#[pymethods]
impl PyWordLevelTrainer {
    #[new]
    #[pyo3(signature = (*, vocab_size = 30000, min_frequency = 0, special_tokens = vec![], show_progress = true))]
    fn new(
        vocab_size: usize,
        min_frequency: u64,
        special_tokens: Vec<TokenInput>,
        show_progress: bool,
    ) -> PyResult<PyClassInitializer<Self>> {
        let trainer = WordLevelTrainer::builder()
            .vocab_size(vocab_size)
            .min_frequency(min_frequency)
            .special_tokens(parse_tokens(special_tokens, true))
            .show_progress(show_progress)
            .build()
            .map_err(|e| to_pyerr(e.to_string().into()))?;
        Ok(PyClassInitializer::from(PyTrainer {
            inner: trainer.into(),
        })
        .add_subclass(PyWordLevelTrainer))
    }
}

#[pymodule(gil_used = false)]
pub mod trainers {
    #[pymodule_export]
    pub use super::{
        PyBpeTrainer, PyTrainer, PyUnigramTrainer, PyWordLevelTrainer, PyWordPieceTrainer,
    };
}
