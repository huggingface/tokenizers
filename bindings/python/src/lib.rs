#![warn(clippy::all)]

pub mod added_token;
pub mod detached_lock;
pub mod error;
pub mod models;
pub mod normalizers;
pub mod pre_tokenizers;
pub mod tokenizer;
pub mod trainers;

use pyo3::prelude::*;

/// Components repr as their tokenizer.json serialization: compact, and always
/// in sync with what `Tokenizer.save` writes.
pub fn component_repr<T: serde::Serialize>(component: &T) -> String {
    serde_json::to_string(component).unwrap_or_else(|_| "<unserializable>".to_owned())
}

// Forked children of a process that used our rayon threads would inherit a
// poisoned thread pool; disable parallelism there unless the user opted in
// explicitly through TOKENIZERS_PARALLELISM.
#[cfg(target_family = "unix")]
extern "C" fn child_after_fork() {
    use std::sync::atomic::Ordering;
    use tk_encode::utils::parallelism::{is_parallelism_configured, set_parallelism};
    if crate::tokenizer::USED_PARALLELISM.load(Ordering::SeqCst) && !is_parallelism_configured() {
        set_parallelism(false);
    }
}

/// Fast tokenizers: turn text into the token ids models consume.
/// Start with `Tokenizer`.
#[pymodule(gil_used = false)]
pub mod _native {
    use super::*;

    #[pymodule_export]
    pub use super::added_token::PyAddedToken;
    #[pymodule_export]
    pub use super::error::TokenizersError;
    #[pymodule_export]
    pub use super::tokenizer::PyTokenizer;

    #[pymodule_export]
    pub use super::models::models;
    #[pymodule_export]
    pub use super::normalizers::normalizers;
    #[pymodule_export]
    pub use super::pre_tokenizers::pre_tokenizers;
    #[pymodule_export]
    pub use super::trainers::trainers;

    #[allow(non_upper_case_globals)]
    #[pymodule_export]
    pub const __version__: &str = env!("CARGO_PKG_VERSION");

    #[pymodule_init]
    fn init(_m: &Bound<'_, PyModule>) -> PyResult<()> {
        #[cfg(target_family = "unix")]
        {
            use std::sync::Once;
            static REGISTER_FORK_CALLBACK: Once = Once::new();
            REGISTER_FORK_CALLBACK.call_once(|| unsafe {
                libc::pthread_atfork(None, None, Some(child_after_fork));
            });
        }
        Ok(())
    }
}
