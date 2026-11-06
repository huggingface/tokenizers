//! Process-wide threading knobs.

use pyo3::prelude::*;
use tk::parallelism;

/// Sets the most threads one encode uses, the calling thread included, for the whole process.
///
/// `1` encodes serially; `0` restores the default, every available core.
///
/// A value set here always wins. Without one, a multiprocessing worker runs serially, since it
/// can't know how many siblings share the cores. Forked workers inherit a value set in the parent;
/// spawn and forkserver workers start fresh, so set it in each worker, e.g.
/// `multiprocessing.Pool(initializer=tokenizers.set_max_threads, initargs=(4,))`.
///
/// Args:
///     max_threads: int
#[pyfunction]
pub fn set_max_threads(max_threads: usize) {
    parallelism::set_max_threads(max_threads);
}

/// The most threads one encode uses, the calling thread included: the value set with
/// `set_max_threads`, else the default.
///
/// Returns:
///     int
#[pyfunction]
pub fn get_max_threads() -> usize {
    parallelism::max_threads()
}
