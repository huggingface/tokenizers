use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};
use tk::vocab::bucket_added_vocabulary::AddedToken as CoreAddedToken;

use crate::repr;

/// A token matched in the text before the model runs, such as `[CLS]` or `<|endoftext|>`.
///
/// Args:
///     content:
///         The text of the token.
///     single_word:
///         Only match the token when it is a whole word, not inside one.
///     lstrip:
///         Include the whitespace to the left of a match in the token.
///     rstrip:
///         Include the whitespace to the right of a match in the token.
///     normalized:
///         Match the token against the normalized text rather than the original text.
///         Defaults to `not special`.
///     special:
///         Skip the token when decoding with `skip_special_tokens`.
#[pyclass(frozen, eq, hash, module = "tokenizers")]
#[derive(PartialEq, Eq, Hash)]
pub struct AddedToken {
    /// The text of the token.
    #[pyo3(get)]
    content: String,
    /// Whether the token only matches a whole word.
    #[pyo3(get)]
    single_word: bool,
    /// Whether a match includes the whitespace to its left.
    #[pyo3(get)]
    lstrip: bool,
    /// Whether a match includes the whitespace to its right.
    #[pyo3(get)]
    rstrip: bool,
    /// Whether the token text is normalized.
    #[pyo3(get)]
    normalized: bool,
    /// Whether decoding with `skip_special_tokens` skips the token.
    #[pyo3(get)]
    special: bool,
}

impl From<CoreAddedToken> for AddedToken {
    fn from(token: CoreAddedToken) -> Self {
        Self {
            content: token.content,
            single_word: token.single_word,
            lstrip: token.lstrip,
            rstrip: token.rstrip,
            normalized: token.normalized,
            special: token.special,
        }
    }
}

#[pymethods]
impl AddedToken {
    #[new]
    #[pyo3(signature = (content, *, single_word=false, lstrip=false, rstrip=false, normalized=None, special=false))]
    fn new(
        content: String,
        single_word: bool,
        lstrip: bool,
        rstrip: bool,
        normalized: Option<bool>,
        special: bool,
    ) -> Self {
        Self {
            content,
            single_word,
            lstrip,
            rstrip,
            normalized: normalized.unwrap_or(!special),
            special,
        }
    }

    /// The token as an `added_tokens` entry without its `id`.
    ///
    /// transformers writes this dict to `tokenizer_config.json`.
    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let state = PyDict::new(py);
        state.set_item("content", &self.content)?;
        state.set_item("single_word", self.single_word)?;
        state.set_item("lstrip", self.lstrip)?;
        state.set_item("rstrip", self.rstrip)?;
        state.set_item("normalized", self.normalized)?;
        state.set_item("special", self.special)?;
        Ok(state)
    }

    /// Pickle rebuilds an `AddedToken` by calling `_unpickle` with `__getstate__()`.
    fn __reduce__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyDict>,))> {
        let state = self.__getstate__(py)?;
        Ok((py.get_type::<Self>().getattr("_unpickle")?, (state,)))
    }

    /// Calls the constructor with `state` as keyword arguments.
    #[staticmethod]
    fn _unpickle<'py>(state: &Bound<'py, PyDict>) -> PyResult<Bound<'py, Self>> {
        let token = state.py().get_type::<Self>().call((), Some(state))?;
        Ok(token.cast_into()?)
    }

    fn __str__(&self) -> &str {
        &self.content
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "AddedToken({}, single_word={}, lstrip={}, rstrip={}, normalized={}, special={})",
            PyString::new(py, &self.content).repr()?,
            repr::python_bool(self.single_word),
            repr::python_bool(self.lstrip),
            repr::python_bool(self.rstrip),
            repr::python_bool(self.normalized),
            repr::python_bool(self.special),
        ))
    }
}
