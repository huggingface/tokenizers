mod models;
mod token;
mod tokenizer;
mod utils;

use pyo3::prelude::*;
use pyo3::wrap_pymodule;

/// Models Module
#[pymodule]
pub fn models(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<models::Model>()?;
    m.add_class::<models::BPE>()?;
    Ok(())
}

/// Tokenizers Module
#[pymodule]
fn tokenizers(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<tokenizer::Tokenizer>()?;
    m.add_wrapped(wrap_pymodule!(models))?;
    Ok(())
}
