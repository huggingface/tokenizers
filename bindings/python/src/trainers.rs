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
                    "vocab_size" => {
                        let size: usize = val.extract()?;
                        trainer.vocab_size = size;
                    }
                    "min_frequency" => {
                        let freq: u32 = val.extract()?;
                        trainer.min_frequency = freq;
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
