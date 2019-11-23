mod models;
mod pre_tokenizers;
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

/// PreTokenizers Module
#[pymodule]
pub fn pre_tokenizers(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<pre_tokenizers::ByteLevel>()?;
    Ok(())
}

/// Tokenizers Module
#[pymodule]
fn tokenizers(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<tokenizer::Tokenizer>()?;
    m.add_wrapped(wrap_pymodule!(models))?;
    m.add_wrapped(wrap_pymodule!(pre_tokenizers))?;
    Ok(())
}
