use pyo3::{PyResult, Python};
use pyo3::prelude::*;

mod encoding;
mod error;
mod token;
mod utils;

pub mod decoders;
pub mod models;
pub mod pre_tokenizers;
pub mod processors;
pub mod tokenizer;
pub mod trainers;


#[pymodule]
fn tokenizers(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<tokenizer::Tokenizer>()?;
    m.add_class::<token::Token>()?;
    m.add_class::<encoding::Encoding>()?;
    Ok(())
}