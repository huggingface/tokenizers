//! Pickling the classes this module exposes.
//!
//! Python pickles an object to move it to another process: `multiprocessing`, a PyTorch
//! `DataLoader` worker, `datasets.map(num_proc=...)`. `copy.deepcopy` goes through pickling too.
//!
//! Every class here is frozen, so the usual `__setstate__`, which writes the state into an object
//! pickle built empty, is out. Each class implements `__reduce__` instead, handing back a
//! callable and the arguments that rebuild the value. That callable is the class itself when its
//! constructor takes exactly those arguments, as `Padding` does, and a private `_unpickle_*`
//! function next to the class otherwise.
//!
//! A class without a `__reduce__` fails `test_every_class_the_module_exports_can_be_pickled`.

use pyo3::PyTypeInfo;
use pyo3::prelude::*;

/// What `__reduce__` hands pickle: something to call, and the arguments to call it with.
pub(crate) type Reduced<Arguments> = (Py<PyAny>, Arguments);

/// The extension module, `module-name` in `pyproject.toml`. The `tokenizers` package that
/// re-exports it holds the classes, not the `_unpickle_*` functions.
const MODULE: &str = "tokenizers.tokenizers";

/// The class itself, for a value whose constructor takes what `__reduce__` hands back.
pub(crate) fn class<T: PyTypeInfo>(py: Python<'_>) -> Py<PyAny> {
    T::type_object(py).into_any().unbind()
}

/// The `_unpickle_*` function called `name`. Pickle stores a function by name and checks that the
/// name resolves back to the same object, so the function has to come off the module itself.
pub(crate) fn rebuilder(py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
    Ok(py.import(MODULE)?.getattr(name)?.unbind())
}
