#![warn(clippy::all)]
#![allow(clippy::upper_case_acronyms)]
// Many false positives with pyo3 it seems &str, and &PyAny get flagged
#![allow(clippy::borrow_deref_ref)]

extern crate tokenizers as tk;

use once_cell::sync::Lazy;
use std::sync::Arc;
use tokio::runtime::Runtime;

// We create a global runtime that will be initialized once when first needed
// This ensures we always have a runtime available for tokio::task::spawn_blocking
static TOKIO_RUNTIME: Lazy<Arc<Runtime>> = Lazy::new(|| {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("Failed to create global Tokio runtime");
    Arc::new(rt)
});
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
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

// For users using multiprocessing in python, it is quite easy to fork the process running
// tokenizers, ending up with a deadlock because we internally make use of multithreading. So
// we register a callback to be called in the event of a fork to disable parallelism.
#[cfg(target_family = "unix")]
static mut REGISTERED_FORK_CALLBACK: bool = false;
#[cfg(target_family = "unix")]
extern "C" fn child_after_fork() {
    use tk::parallelism::*;
    if has_parallelism_been_used() && !is_parallelism_configured() {
        set_parallelism(false);
    }
}

/// Tokenizers Module
#[pymodule]
pub mod tokenizers {
    use super::*;

    #[pymodule_export]
    pub use super::encoding::PyEncoding;
    #[pymodule_export]
    pub use super::token::PyToken;
    #[pymodule_export]
    pub use super::tokenizer::PyAddedToken;
    #[pymodule_export]
    pub use super::tokenizer::PyTokenizer;
    #[pymodule_export]
    pub use super::utils::PyNormalizedString;
    #[pymodule_export]
    pub use super::utils::PyPreTokenizedString;
    #[pymodule_export]
    pub use super::utils::PyRegex;

    #[pymodule_export]
    pub use super::decoders::decoders;
    #[pymodule_export]
    pub use super::models::models;
    #[pymodule_export]
    pub use super::normalizers::normalizers;
    #[pymodule_export]
    pub use super::pre_tokenizers::pre_tokenizers;
    #[pymodule_export]
    pub use super::processors::processors;
    #[pymodule_export]
    pub use super::trainers::trainers;

    #[allow(non_upper_case_globals)]
    #[pymodule_export]
    pub const __version__: &str = env!("CARGO_PKG_VERSION");

    #[pymodule_init]
    fn init(_m: &Bound<'_, PyModule>) -> PyResult<()> {
        let _ = env_logger::try_init_from_env("TOKENIZERS_LOG");

        // Register the fork callback
        #[cfg(target_family = "unix")]
        unsafe {
            if !REGISTERED_FORK_CALLBACK {
                libc::pthread_atfork(None, None, Some(child_after_fork));
                REGISTERED_FORK_CALLBACK = true;
            }
        }

        Ok(())
    }
}
