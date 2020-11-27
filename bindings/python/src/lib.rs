#![warn(clippy::all)]

extern crate tokenizers as tk;

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

// For users using multiprocessing in python, it is quite easy to fork the process running
// tokenizers, ending up with a deadlock because we internaly make use of multithreading. So
// we register a callback to be called in the event of a fork so that we can warn the user.
static mut REGISTERED_FORK_CALLBACK: bool = false;
extern "C" fn child_after_fork() {
    use tk::parallelism::*;
    if has_parallelism_been_used() && !is_parallelism_configured() {
        println!(
            "huggingface/tokenizers: The current process just got forked, after parallelism has \
            already been used. Disabling parallelism to avoid deadlocks..."
        );
        println!("To disable this warning, you can either:");
        println!(
            "\t- Avoid using `tokenizers` before the fork if possible\n\
            \t- Explicitly set the environment variable {}=(true | false)",
            ENV_VARIABLE
        );
        set_parallelism(false);
    }
}

/// Trainers Module
#[pymodule]
fn trainers(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<trainers::PyTrainer>()?;
    m.add_class::<trainers::PyBpeTrainer>()?;
    m.add_class::<trainers::PyWordPieceTrainer>()?;
    m.add_class::<trainers::PyWordLevelTrainer>()?;
    m.add_class::<trainers::PyUnigramTrainer>()?;
    Ok(())
}

/// Models Module
#[pymodule]
fn models(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<models::PyModel>()?;
    m.add_class::<models::PyBPE>()?;
    m.add_class::<models::PyWordPiece>()?;
    m.add_class::<models::PyWordLevel>()?;
    m.add_class::<models::PyUnigram>()?;
    Ok(())
}

/// PreTokenizers Module
#[pymodule]
fn pre_tokenizers(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<pre_tokenizers::PyPreTokenizer>()?;
    m.add_class::<pre_tokenizers::PyByteLevel>()?;
    m.add_class::<pre_tokenizers::PyWhitespace>()?;
    m.add_class::<pre_tokenizers::PyWhitespaceSplit>()?;
    m.add_class::<pre_tokenizers::PySplit>()?;
    m.add_class::<pre_tokenizers::PyBertPreTokenizer>()?;
    m.add_class::<pre_tokenizers::PyMetaspace>()?;
    m.add_class::<pre_tokenizers::PyCharDelimiterSplit>()?;
    m.add_class::<pre_tokenizers::PyPunctuation>()?;
    m.add_class::<pre_tokenizers::PySequence>()?;
    m.add_class::<pre_tokenizers::PyDigits>()?;
    m.add_class::<pre_tokenizers::PyUnicodeScripts>()?;
    Ok(())
}

/// Decoders Module
#[pymodule]
fn decoders(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<decoders::PyDecoder>()?;
    m.add_class::<decoders::PyByteLevelDec>()?;
    m.add_class::<decoders::PyWordPieceDec>()?;
    m.add_class::<decoders::PyMetaspaceDec>()?;
    m.add_class::<decoders::PyBPEDecoder>()?;
    Ok(())
}

/// Processors Module
#[pymodule]
fn processors(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<processors::PyPostProcessor>()?;
    m.add_class::<processors::PyBertProcessing>()?;
    m.add_class::<processors::PyRobertaProcessing>()?;
    m.add_class::<processors::PyByteLevel>()?;
    m.add_class::<processors::PyTemplateProcessing>()?;
    Ok(())
}

/// Normalizers Module
#[pymodule]
fn normalizers(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<normalizers::PyNormalizer>()?;
    m.add_class::<normalizers::PyBertNormalizer>()?;
    m.add_class::<normalizers::PyNFD>()?;
    m.add_class::<normalizers::PyNFKD>()?;
    m.add_class::<normalizers::PyNFC>()?;
    m.add_class::<normalizers::PyNFKC>()?;
    m.add_class::<normalizers::PySequence>()?;
    m.add_class::<normalizers::PyLowercase>()?;
    m.add_class::<normalizers::PyStrip>()?;
    m.add_class::<normalizers::PyStripAccents>()?;
    m.add_class::<normalizers::PyNmt>()?;
    m.add_class::<normalizers::PyPrecompiled>()?;
    m.add_class::<normalizers::PyReplace>()?;
    Ok(())
}

/// Tokenizers Module
#[pymodule]
fn tokenizers(_py: Python, m: &PyModule) -> PyResult<()> {
    env_logger::init_from_env("TOKENIZERS_LOG");

    // Register the fork callback
    #[cfg(target_family = "unix")]
    unsafe {
        if !REGISTERED_FORK_CALLBACK {
            libc::pthread_atfork(None, None, Some(child_after_fork));
            REGISTERED_FORK_CALLBACK = true;
        }
    }

    m.add_class::<tokenizer::PyTokenizer>()?;
    m.add_class::<tokenizer::PyAddedToken>()?;
    m.add_class::<token::PyToken>()?;
    m.add_class::<encoding::PyEncoding>()?;
    m.add_class::<utils::PyRegex>()?;
    m.add_class::<utils::PyNormalizedString>()?;
    m.add_class::<utils::PyPreTokenizedString>()?;
    m.add_wrapped(wrap_pymodule!(models))?;
    m.add_wrapped(wrap_pymodule!(pre_tokenizers))?;
    m.add_wrapped(wrap_pymodule!(decoders))?;
    m.add_wrapped(wrap_pymodule!(processors))?;
    m.add_wrapped(wrap_pymodule!(normalizers))?;
    m.add_wrapped(wrap_pymodule!(trainers))?;
    Ok(())
}
