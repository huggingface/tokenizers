//! Argument and return types whose type hints pyo3 cannot derive

use std::collections::BTreeMap;
use std::convert::Infallible;

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyTypeError;
use pyo3::inspect::PyStaticExpr;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyString};
use pyo3::{Borrowed, PyTypeInfo, type_hint_identifier, type_hint_subscript, type_hint_union};

use crate::added_token::AddedToken;
use crate::encoding::Encoding;

/// The `ids` argument of `Tokenizer.decode` and `Tokenizer.decode_tokens`, typed
/// `Encoding | Sequence[int] | Buffer`.
///
/// An `Encoding` is read in place, no copy.
/// A 1-D buffer of `uint32`, such as a numpy array, is copied in one block.
/// Any other sequence is copied into a `Vec<u32>` one element at a time.
pub enum TokenIds<'py> {
    Encoding(Bound<'py, Encoding>),
    Copied(Vec<u32>),
}

impl<'py> FromPyObject<'_, 'py> for TokenIds<'py> {
    type Error = PyErr;

    const INPUT_TYPE: PyStaticExpr = type_hint_union!(
        <Bound<'static, Encoding> as FromPyObject<'static, 'static>>::INPUT_TYPE,
        type_hint_subscript!(
            type_hint_identifier!("collections.abc", "Sequence"),
            type_hint_identifier!("builtins", "int")
        ),
        // `collections.abc.Buffer` needs Python 3.12. A numpy array is a buffer but not a `Sequence`.
        type_hint_identifier!("typing_extensions", "Buffer")
    );

    fn extract(obj: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(encoding) = obj.extract::<Bound<'py, Encoding>>() {
            return Ok(Self::Encoding(encoding));
        }
        if let Ok(buffer) = PyBuffer::<u32>::get(&obj)
            && buffer.dimensions() == 1
        {
            return buffer.to_vec(obj.py()).map(Self::Copied);
        }
        obj.extract::<Vec<u32>>().map(Self::Copied)
    }
}

impl TokenIds<'_> {
    pub fn as_slice(&self) -> &[u32] {
        match self {
            Self::Encoding(encoding) => encoding.get().ids(),
            Self::Copied(ids) => ids,
        }
    }
}

/// The `token` argument of `Tokenizer.from_pretrained`, typed `str | bool`.
///
/// Handed to `huggingface_hub` as is: it reads `True` as the stored token and `False` as no token.
pub struct Token<'py>(pub Bound<'py, PyAny>);

impl<'py> FromPyObject<'_, 'py> for Token<'py> {
    type Error = PyErr;

    const INPUT_TYPE: PyStaticExpr = type_hint_union!(PyString::TYPE_HINT, PyBool::TYPE_HINT);

    fn extract(obj: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
        if obj.is_instance_of::<PyString>() || obj.is_instance_of::<PyBool>() {
            return Ok(Self(obj.to_owned()));
        }
        Err(PyTypeError::new_err(format!(
            "token must be a str or a bool, not {}",
            obj.get_type().name()?
        )))
    }
}

/// The dict `CompiledTokenizer.added_tokens` returns, typed `dict[int, AddedToken]`.
///
/// A bare [`PyDict`] in a signature is typed `dict`.
pub struct AddedTokens<'py>(pub Bound<'py, PyDict>);

impl<'py> IntoPyObject<'py> for AddedTokens<'py> {
    type Target = PyDict;
    type Output = Bound<'py, PyDict>;
    type Error = Infallible;

    const OUTPUT_TYPE: PyStaticExpr =
        <BTreeMap<u32, AddedToken> as IntoPyObject<'static>>::OUTPUT_TYPE;

    fn into_pyobject(self, _py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(self.0)
    }
}

/// What `Encoding.ids_buffer` and the other buffer getters return, typed `Buffer`.
///
/// The stubs promise only the buffer protocol, so the class behind it can change.
pub struct U32Buffer<'py>(pub Bound<'py, PyAny>);

impl<'py> IntoPyObject<'py> for U32Buffer<'py> {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = Infallible;

    const OUTPUT_TYPE: PyStaticExpr = type_hint_identifier!("typing_extensions", "Buffer");

    fn into_pyobject(self, _py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(self.0)
    }
}
