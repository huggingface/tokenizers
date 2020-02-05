mod decoders;
mod encoding;
mod error;
mod models;
mod normalizers;
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
    m.add_class::<trainers::WordPieceTrainer>()?;
    Ok(())
}

/// Models Module
#[pymodule]
fn models(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<models::Model>()?;
    m.add_class::<models::BPE>()?;
    m.add_class::<models::WordPiece>()?;
    m.add_class::<models::WordLevel>()?;
    Ok(())
}

/// PreTokenizers Module
#[pymodule]
fn pre_tokenizers(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<pre_tokenizers::PreTokenizer>()?;
    m.add_class::<pre_tokenizers::ByteLevel>()?;
    m.add_class::<pre_tokenizers::Whitespace>()?;
    m.add_class::<pre_tokenizers::WhitespaceSplit>()?;
    m.add_class::<pre_tokenizers::BertPreTokenizer>()?;
    m.add_class::<pre_tokenizers::Metaspace>()?;
    m.add_class::<pre_tokenizers::CharDelimiterSplit>()?;
    Ok(())
}

/// Decoders Module
#[pymodule]
fn decoders(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<decoders::Decoder>()?;
    m.add_class::<decoders::ByteLevel>()?;
    m.add_class::<decoders::WordPiece>()?;
    m.add_class::<decoders::Metaspace>()?;
    m.add_class::<decoders::BPEDecoder>()?;
    Ok(())
}

/// Processors Module
#[pymodule]
fn processors(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<processors::PostProcessor>()?;
    m.add_class::<processors::BertProcessing>()?;
    m.add_class::<processors::RobertaProcessing>()?;
    Ok(())
}

/// Normalizers Module
#[pymodule]
fn normalizers(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<normalizers::Normalizer>()?;
    m.add_class::<normalizers::BertNormalizer>()?;
    m.add_class::<normalizers::NFD>()?;
    m.add_class::<normalizers::NFKD>()?;
    m.add_class::<normalizers::NFC>()?;
    m.add_class::<normalizers::NFKC>()?;
    m.add_class::<normalizers::Sequence>()?;
    m.add_class::<normalizers::Lowercase>()?;
    Ok(())
}

/// Tokenizers Module
#[pymodule]
fn tokenizers(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<tokenizer::Tokenizer>()?;
    m.add_class::<encoding::Encoding>()?;
    m.add_wrapped(wrap_pymodule!(models))?;
    m.add_wrapped(wrap_pymodule!(pre_tokenizers))?;
    m.add_wrapped(wrap_pymodule!(decoders))?;
    m.add_wrapped(wrap_pymodule!(processors))?;
    m.add_wrapped(wrap_pymodule!(normalizers))?;
    m.add_wrapped(wrap_pymodule!(trainers))?;
    Ok(())
}
