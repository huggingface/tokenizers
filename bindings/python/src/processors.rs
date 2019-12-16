extern crate tokenizers as tk;

use super::error::PyError;
use super::utils::Container;
use pyo3::prelude::*;
use pyo3::types::*;
use tk::utils::TruncationStrategy;

#[pyclass(dict)]
pub struct PostProcessor {
    pub processor: Container<dyn tk::tokenizer::PostProcessor + Sync>,
}

#[pyclass]
pub struct BertProcessing {}
#[pymethods]
impl BertProcessing {
    #[staticmethod]
    #[args(kwargs = "**")]
    fn new(
        sep: (String, u32),
        cls: (String, u32),
        kwargs: Option<&PyDict>,
    ) -> PyResult<PostProcessor> {
        let mut max_len = 512;
        let mut trunc_strategy = tk::utils::TruncationStrategy::LongestFirst;
        let mut trunc_stride = 0;

        if let Some(kwargs) = kwargs {
            for (key, val) in kwargs {
                let key: &str = key.extract()?;
                match key {
                    "max_len" => max_len = val.extract()?,
                    "trunc_stride" => trunc_stride = val.extract()?,
                    "trunc_strategy" => {
                        let strategy: &str = val.extract()?;
                        trunc_strategy = match strategy {
                            "longest_first" => Ok(TruncationStrategy::LongestFirst),
                            "only_first" => Ok(TruncationStrategy::OnlyFirst),
                            "only_second" => Ok(TruncationStrategy::OnlySecond),
                            other => Err(PyError(format!(
                                "Unknown `trunc_strategy`: `{}`. Use \
                                 one of `longest_first`, `only_first` or `only_second`",
                                other
                            ))
                            .into_pyerr()),
                        }?;
                    }
                    _ => println!("Ignored unknown kwargs option {}", key),
                }
            }
        }

        Ok(PostProcessor {
            processor: Container::Owned(Box::new(tk::processors::bert::BertProcessing::new(
                max_len,
                trunc_strategy,
                trunc_stride,
                sep,
                cls,
            ))),
        })
    }
}
