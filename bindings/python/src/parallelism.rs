//! How many threads one encode uses.

use pyo3::prelude::*;
use tk::parallelism;

/// Sets the most threads one encode uses, the calling thread included, for the whole process.
///
/// `1` encodes serially, and `0` restores the default: every available core. A forked process
/// always encodes serially, and so does every encode while `TOKENIZERS_PARALLELISM` is `false`.
///
/// Args:
///     max_threads: int
#[pyfunction]
pub(crate) fn set_max_threads(max_threads: usize) {
    parallelism::set_max_threads(max_threads);
}

/// The most threads one encode uses, the calling thread included.
///
/// Returns:
///     int
#[pyfunction]
pub(crate) fn get_max_threads(py: Python<'_>) -> PyResult<usize> {
    if serial_by_env(py)? {
        return Ok(1);
    }
    Ok(parallelism::max_threads())
}

/// Whether `TOKENIZERS_PARALLELISM` holds an off value, which makes every encode serial.
///
/// Reads Python's `os.environ` while holding the GIL: libraries set the variable at run time, and
/// C `getenv` would race with Python's `putenv` on another thread.
pub(crate) fn serial_by_env(py: Python<'_>) -> PyResult<bool> {
    let value: Option<String> = py
        .import("os")?
        .getattr("environ")?
        .call_method1("get", ("TOKENIZERS_PARALLELISM",))?
        .extract()?;
    let Some(value) = value else {
        return Ok(false);
    };
    let off = ["", "off", "false", "f", "no", "n", "0"];
    Ok(off.contains(&value.to_ascii_lowercase().as_str()))
}
