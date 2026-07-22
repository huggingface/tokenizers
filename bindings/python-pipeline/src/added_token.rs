use pyo3::prelude::*;
use tk_encode::tokenizer::AddedToken;

/// A token added on top of the model's vocabulary, with its matching options.
#[pyclass(
    frozen,
    from_py_object,
    name = "AddedToken",
    module = "tokenizers_pipeline"
)]
#[derive(Clone)]
pub struct PyAddedToken {
    pub inner: AddedToken,
}

#[pymethods]
impl PyAddedToken {
    #[new]
    #[pyo3(signature = (content, *, single_word = false, lstrip = false, rstrip = false, normalized = None, special = false))]
    fn new(
        content: String,
        single_word: bool,
        lstrip: bool,
        rstrip: bool,
        normalized: Option<bool>,
        special: bool,
    ) -> Self {
        let inner = AddedToken::from(content, special)
            .single_word(single_word)
            .lstrip(lstrip)
            .rstrip(rstrip)
            .normalized(normalized.unwrap_or(!special));
        Self { inner }
    }

    #[getter]
    fn content(&self) -> &str {
        &self.inner.content
    }

    #[getter]
    fn single_word(&self) -> bool {
        self.inner.single_word
    }

    #[getter]
    fn lstrip(&self) -> bool {
        self.inner.lstrip
    }

    #[getter]
    fn rstrip(&self) -> bool {
        self.inner.rstrip
    }

    #[getter]
    fn normalized(&self) -> bool {
        self.inner.normalized
    }

    #[getter]
    fn special(&self) -> bool {
        self.inner.special
    }

    fn __repr__(&self) -> String {
        format!(
            "AddedToken({:?}, single_word={}, lstrip={}, rstrip={}, normalized={}, special={})",
            self.inner.content,
            self.inner.single_word,
            self.inner.lstrip,
            self.inner.rstrip,
            self.inner.normalized,
            self.inner.special
        )
    }
}

/// A `str | AddedToken` argument.
#[derive(FromPyObject)]
pub enum TokenInput {
    Str(String),
    Token(PyAddedToken),
}

/// Plain strings become tokens with `special=special_default` (and
/// `normalized=!special_default`, matching v1).
pub fn parse_tokens(items: Vec<TokenInput>, special_default: bool) -> Vec<AddedToken> {
    items
        .into_iter()
        .map(|item| match item {
            TokenInput::Str(content) => AddedToken::from(content, special_default),
            TokenInput::Token(token) => {
                let mut inner = token.inner;
                if special_default {
                    inner.special = true;
                }
                inner
            }
        })
        .collect()
}
