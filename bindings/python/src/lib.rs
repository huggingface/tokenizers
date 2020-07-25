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
    Ok(())
}

/// Models Module
#[pymodule]
fn models(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<models::PyModel>()?;
    m.add_class::<models::PyBPE>()?;
    m.add_class::<models::PyWordPiece>()?;
    m.add_class::<models::PyWordLevel>()?;
    Ok(())
}

/// PreTokenizers Module
#[pymodule]
fn pre_tokenizers(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<pre_tokenizers::PyPreTokenizer>()?;
    m.add_class::<pre_tokenizers::PyByteLevel>()?;
    m.add_class::<pre_tokenizers::PyWhitespace>()?;
    m.add_class::<pre_tokenizers::PyWhitespaceSplit>()?;
    m.add_class::<pre_tokenizers::PyBertPreTokenizer>()?;
    m.add_class::<pre_tokenizers::PyMetaspace>()?;
    m.add_class::<pre_tokenizers::PyCharDelimiterSplit>()?;
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
    m.add_class::<processors::ByteLevel>()?;
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
    Ok(())
}

/// Tokenizers Module
#[pymodule]
fn tokenizers(_py: Python, m: &PyModule) -> PyResult<()> {
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
    m.add_class::<encoding::PyEncoding>()?;
    m.add_wrapped(wrap_pymodule!(models))?;
    m.add_wrapped(wrap_pymodule!(pre_tokenizers))?;
    m.add_wrapped(wrap_pymodule!(decoders))?;
    m.add_wrapped(wrap_pymodule!(processors))?;
    m.add_wrapped(wrap_pymodule!(normalizers))?;
    m.add_wrapped(wrap_pymodule!(trainers))?;
    Ok(())
}
