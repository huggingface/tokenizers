//! Process-wide threading knobs.

use pyo3::prelude::*;
use tk::parallelism;

/// Sets the most threads one encode uses, the calling thread included, for the whole process.
///
/// `1` encodes serially; `0` restores the default, every available core.
/// A multiprocessing worker starts at `1`, since it can't know how many siblings share the cores.
///
/// Args:
///     max_threads: int
#[pyfunction]
pub fn set_max_threads(max_threads: usize) {
    parallelism::set_max_threads(max_threads);
}

/// The most threads one encode uses, the calling thread included.
///
/// Returns:
///     int
#[pyfunction]
pub fn get_max_threads() -> usize {
    parallelism::max_threads()
}
