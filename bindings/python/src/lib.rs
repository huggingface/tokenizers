mod decoders;
mod encoding;
mod error;
mod models;
mod pre_tokenizers;
mod processors;
mod token;
mod tokenizer;
mod trainers;
mod utils;

use pyo3::prelude::*;
use pyo3::wrap_pymodule;

/// Trainers Module
#[pymodule]
fn trainers(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<trainers::Trainer>()?;
    m.add_class::<trainers::BpeTrainer>()?;
    Ok(())
}

/// Models Module
#[pymodule]
fn models(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<models::Model>()?;
    m.add_class::<models::BPE>()?;
    m.add_class::<models::WordPiece>()?;
    Ok(())
}

/// PreTokenizers Module
#[pymodule]
fn pre_tokenizers(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<pre_tokenizers::PreTokenizer>()?;
    m.add_class::<pre_tokenizers::ByteLevel>()?;
    m.add_class::<pre_tokenizers::BertPreTokenizer>()?;
    Ok(())
}

/// Decoders Module
#[pymodule]
fn decoders(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<decoders::Decoder>()?;
    m.add_class::<decoders::ByteLevel>()?;
    m.add_class::<decoders::WordPiece>()?;
    Ok(())
}

/// Processors Module
#[pymodule]
fn processors(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<processors::PostProcessor>()?;
    m.add_class::<processors::BertProcessing>()?;
    Ok(())
}

/// Tokenizers Module
#[pymodule]
fn tokenizers(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<tokenizer::Tokenizer>()?;
    m.add_wrapped(wrap_pymodule!(models))?;
    m.add_wrapped(wrap_pymodule!(pre_tokenizers))?;
    m.add_wrapped(wrap_pymodule!(decoders))?;
    m.add_wrapped(wrap_pymodule!(processors))?;
    m.add_wrapped(wrap_pymodule!(trainers))?;
    Ok(())
}
