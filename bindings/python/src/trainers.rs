extern crate tokenizers as tk;

use super::utils::Container;
use pyo3::prelude::*;
use pyo3::types::*;

#[pyclass]
pub struct Trainer {
    pub trainer: Container<dyn tk::tokenizer::Trainer>,
}

#[pyclass]
pub struct BpeTrainer {}
#[pymethods]
impl BpeTrainer {
    /// new(/vocab_size, min_frequency)
    /// --
    ///
    /// Create a new BpeTrainer with the given configuration
    #[staticmethod]
    #[args(kwargs = "**")]
    pub fn new(kwargs: Option<&PyDict>) -> PyResult<Trainer> {
        let mut trainer = tk::models::bpe::BpeTrainer::default();
        if let Some(kwargs) = kwargs {
            for (key, val) in kwargs {
                let key: &str = key.extract()?;
                match key {
                    "vocab_size" => trainer.vocab_size = val.extract()?,
                    "min_frequency" => trainer.min_frequency = val.extract()?,
                    "show_progress" => trainer.show_progress = val.extract()?,
                    "special_tokens" => trainer.special_tokens = val.extract()?,
                    "limit_alphabet" => trainer.limit_alphabet = val.extract()?,
                    "initial_alphabet" => {
                        let alphabet: Vec<String> = val.extract()?;
                        trainer.initial_alphabet = alphabet
                            .into_iter()
                            .map(|s| s.chars().nth(0))
                            .filter(|c| c.is_some())
                            .map(|c| c.unwrap())
                            .collect();
                    }
                    _ => println!("Ignored unknown kwargs option {}", key),
                };
            }
        }

        Ok(Trainer {
            trainer: Container::Owned(Box::new(trainer)),
        })
    }
}
