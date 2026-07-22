use pyo3::prelude::*;
use tk_train::trainers::{
    BpeTrainer, TrainerWrapper, UnigramTrainer, WordLevelTrainer, WordPieceTrainer,
};

use crate::added_token::{TokenInput, parse_tokens};
use crate::error::to_pyerr;

/// Base class for all trainers.
///
/// A trainer is the recipe for learning a model's vocabulary from text; pass
/// one to `Tokenizer.train` or `train_from_iterator`. Trainers are plain
/// configuration values — training copies them and writes nothing back.
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

/// Learns a BPE vocabulary: keeps merging the most frequent pair until
/// `vocab_size` is reached, ignoring pairs seen fewer than `min_frequency`
/// times. `special_tokens` get the first ids. `limit_alphabet` caps how many
/// distinct characters are kept; `initial_alphabet` forces characters in even
/// if the data never shows them; `max_token_length` caps merged token length.
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

/// Learns a WordPiece vocabulary. Same knobs as `BpeTrainer`, plus the
/// continuation prefix ("##" by default).
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

/// Learns a Unigram vocabulary: starts from a large candidate set and prunes
/// it by `shrinking_factor` each round until `vocab_size` pieces remain.
/// `unk_token` names the fallback piece for unknown characters.
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

/// Learns a WordLevel vocabulary: the `vocab_size` most frequent words,
/// keeping only those seen at least `min_frequency` times.
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

/// Recipes for learning a vocabulary from text.
#[pymodule(gil_used = false)]
pub mod trainers {
    #[pymodule_export]
    pub use super::{
        PyBpeTrainer, PyTrainer, PyUnigramTrainer, PyWordLevelTrainer, PyWordPieceTrainer,
    };
}
