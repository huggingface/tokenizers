use std::hash::{Hash, Hasher};

use pyo3::prelude::*;
use pyo3::types::PyDict;
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
#[derive(PartialEq, Eq)]
pub struct AddedToken(CoreAddedToken);

// Equal tokens have equal content, so hashing the content alone agrees with `==`.
impl Hash for AddedToken {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.content.hash(state);
    }
}

impl From<CoreAddedToken> for AddedToken {
    fn from(token: CoreAddedToken) -> Self {
        Self(token)
    }
}

/// 0.23's `AddedToken.__getstate__`: the token's `added_tokens` entry without its `id`, as a
/// dict. transformers' `save_pretrained` writes it to `tokenizer_config.json`.
#[derive(FromPyObject, IntoPyObject)]
#[pyo3(from_item_all)]
struct State {
    content: String,
    single_word: bool,
    lstrip: bool,
    rstrip: bool,
    normalized: bool,
    special: bool,
}

/// A token passed to `Tokenizer.add_tokens` or `Tokenizer.add_special_tokens`: its content, or
/// an `AddedToken`.
#[derive(FromPyObject)]
pub enum TokenToAdd<'py> {
    Content(String),
    Token(PyRef<'py, AddedToken>),
}

impl TokenToAdd<'_> {
    /// The token to add. `special` marks it special, and a bare content gets the default flags of
    /// `AddedToken(content, special=special)`.
    pub fn into_added_token(self, special: bool) -> CoreAddedToken {
        match self {
            Self::Content(content) => CoreAddedToken::from(content, special),
            Self::Token(token) => CoreAddedToken {
                special: token.0.special || special,
                ..token.0.clone()
            },
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
        CoreAddedToken {
            content,
            single_word,
            lstrip,
            rstrip,
            normalized: normalized.unwrap_or(!special),
            special,
        }
        .into()
    }

    /// The text of the token.
    #[getter]
    fn content(&self) -> &str {
        &self.0.content
    }

    /// Whether the token only matches a whole word.
    #[getter]
    fn single_word(&self) -> bool {
        self.0.single_word
    }

    /// Whether a match includes the whitespace to its left.
    #[getter]
    fn lstrip(&self) -> bool {
        self.0.lstrip
    }

    /// Whether a match includes the whitespace to its right.
    #[getter]
    fn rstrip(&self) -> bool {
        self.0.rstrip
    }

    /// Whether the token text is normalized.
    #[getter]
    fn normalized(&self) -> bool {
        self.0.normalized
    }

    /// Whether decoding with `skip_special_tokens` skips the token.
    #[getter]
    fn special(&self) -> bool {
        self.0.special
    }

    /// The token's `added_tokens` entry without its `id`, as a dict.
    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let token = &self.0;
        let state = State {
            content: token.content.clone(),
            single_word: token.single_word,
            lstrip: token.lstrip,
            rstrip: token.rstrip,
            normalized: token.normalized,
            special: token.special,
        };
        Ok(state.into_pyobject(py)?.cast_into()?)
    }

    /// Pickle rebuilds an `AddedToken` by calling `_unpickle` with `__getstate__()`.
    fn __reduce__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyDict>,))> {
        Ok((
            py.get_type::<Self>().getattr("_unpickle")?,
            (self.__getstate__(py)?,),
        ))
    }

    /// Rebuilds an `AddedToken` from its `__getstate__`.
    #[staticmethod]
    fn _unpickle(state: &Bound<'_, PyDict>) -> PyResult<Self> {
        let State {
            content,
            single_word,
            lstrip,
            rstrip,
            normalized,
            special,
        } = state.extract()?;
        Ok(Self::new(
            content,
            single_word,
            lstrip,
            rstrip,
            Some(normalized),
            special,
        ))
    }

    fn __str__(&self) -> &str {
        self.content()
    }

    fn __repr__(&self) -> String {
        repr::added_token(&self.0)
    }
}
