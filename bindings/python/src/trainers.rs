extern crate tokenizers as tk;

use std::collections::HashMap;

use crate::models::Model;
use crate::utils::Container;
use pyo3::exceptions::Exception;
use pyo3::prelude::*;
use pyo3::types::*;

#[pyclass]
pub struct Trainer {
    pub trainer: Container<dyn tk::tokenizer::Trainer>,
}

#[pymethods]
impl Trainer {
    pub fn train(&self, word_counts: &PyDict) -> PyResult<Model> {
        let mut freqs: HashMap<String, u32> = HashMap::new();
        for (key, value) in word_counts {
            let key: &str = key.extract()?;
            let value: u32 = value.extract()?;
            freqs.insert(key.into(), value);
        }
        let (model, _) = self.trainer.execute(|t| {
            t.train(freqs)
                .map_err(|e| Exception::py_err(format!("{}", e)))
        })?;
        Ok(Model {
            model: Container::Owned(model),
        })
    }
}

#[pyclass]
pub struct BpeTrainer {}
#[pymethods]
impl BpeTrainer {
    /// new(/ vocab_size, min_frequency)
    /// --
    ///
    /// Create a new BpeTrainer with the given configuration
    #[staticmethod]
    #[args(kwargs = "**")]
    pub fn new(kwargs: Option<&PyDict>) -> PyResult<Trainer> {
        let mut builder = tk::models::bpe::BpeTrainer::builder();
        if let Some(kwargs) = kwargs {
            for (key, val) in kwargs {
                let key: &str = key.extract()?;
                match key {
                    "vocab_size" => builder = builder.vocab_size(val.extract()?),
                    "min_frequency" => builder = builder.min_frequency(val.extract()?),
                    "show_progress" => builder = builder.show_progress(val.extract()?),
                    "special_tokens" => builder = builder.special_tokens(val.extract()?),
                    "limit_alphabet" => builder = builder.limit_alphabet(val.extract()?),
                    "initial_alphabet" => {
                        let alphabet: Vec<String> = val.extract()?;
                        builder = builder.initial_alphabet(
                            alphabet
                                .into_iter()
                                .map(|s| s.chars().nth(0))
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

        Ok(Trainer {
            trainer: Container::Owned(Box::new(builder.build())),
        })
    }
}

#[pyclass]
pub struct WordPieceTrainer {}
#[pymethods]
impl WordPieceTrainer {
    /// new(/ vocab_size, min_frequency)
    /// --
    ///
    /// Create a new BpeTrainer with the given configuration
    #[staticmethod]
    #[args(kwargs = "**")]
    pub fn new(kwargs: Option<&PyDict>) -> PyResult<Trainer> {
        let mut builder = tk::models::wordpiece::WordPieceTrainer::builder();
        if let Some(kwargs) = kwargs {
            for (key, val) in kwargs {
                let key: &str = key.extract()?;
                match key {
                    "vocab_size" => builder = builder.vocab_size(val.extract()?),
                    "min_frequency" => builder = builder.min_frequency(val.extract()?),
                    "show_progress" => builder = builder.show_progress(val.extract()?),
                    "special_tokens" => builder = builder.special_tokens(val.extract()?),
                    "limit_alphabet" => builder = builder.limit_alphabet(val.extract()?),
                    "initial_alphabet" => {
                        let alphabet: Vec<String> = val.extract()?;
                        builder = builder.initial_alphabet(
                            alphabet
                                .into_iter()
                                .map(|s| s.chars().nth(0))
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

        Ok(Trainer {
            trainer: Container::Owned(Box::new(builder.build())),
        })
    }
}
