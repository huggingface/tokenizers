//! Python bindings for Hugging Face's Tokenizers rust library
//!
//! Encode text to token ids and decode token ids back to text
mod encoding;
mod error;
mod input;
mod options;
mod repr;
mod tokenizer;
mod type_hints;

use pyo3::prelude::*;

/// Sets the number of threads used to encode.
///
/// 0 restores the default: ``RAYON_NUM_THREADS`` if set, otherwise all available cores.
///
/// Args:
///     n: The number of threads.
#[pyfunction]
fn set_num_threads(n: usize) {
    tk_encode::parallelism::set_num_threads(n);
}

/// The number of threads used to encode.
///
/// Returns:
///     int
#[pyfunction]
fn num_threads() -> usize {
    tk_encode::parallelism::num_threads()
}

#[pyo3::pymodule]
mod tokenizers {
    #[allow(non_upper_case_globals)]
    #[pymodule_export]
    const __version__: &str = env!("CARGO_PKG_VERSION");

    #[pymodule_export]
    use crate::encoding::Encoding;
    #[pymodule_export]
    use crate::options::Padding;
    #[pymodule_export]
    use crate::options::Truncation;
    #[pymodule_export]
    use crate::tokenizer::Tokenizer;
    #[pymodule_export]
    use crate::{num_threads, set_num_threads};

    // TODO: bind trainers
}
