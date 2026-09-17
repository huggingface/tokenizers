//! Argument and return types whose type hints pyo3 cannot derive

use std::convert::Infallible;

use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyTypeError;
use pyo3::inspect::PyStaticExpr;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyString};
use pyo3::{Borrowed, PyTypeInfo, type_hint_identifier, type_hint_subscript, type_hint_union};

/// New type to implement PyO3 introspection traits on
pub struct U32Array<'py>(pub Bound<'py, PyArray1<u32>>);

impl<'py> IntoPyObject<'py> for U32Array<'py> {
    type Target = PyArray1<u32>;
    type Output = Bound<'py, PyArray1<u32>>;
    type Error = Infallible;

    const OUTPUT_TYPE: PyStaticExpr = type_hint_subscript!(
        type_hint_identifier!("numpy.typing", "NDArray"),
        type_hint_identifier!("numpy", "uint32")
    );

    fn into_pyobject(self, _py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(self.0)
    }
}

/// The same, for the `(rows, stride)` array a padded batch reads as.
pub struct U32Array2<'py>(pub Bound<'py, PyArray2<u32>>);

impl<'py> IntoPyObject<'py> for U32Array2<'py> {
    type Target = PyArray2<u32>;
    type Output = Bound<'py, PyArray2<u32>>;
    type Error = Infallible;

    const OUTPUT_TYPE: PyStaticExpr = type_hint_subscript!(
        type_hint_identifier!("numpy.typing", "NDArray"),
        type_hint_identifier!("numpy", "uint32")
    );

    fn into_pyobject(self, _py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(self.0)
    }
}

/// Inputs for `Tokenizer.decode`, typed `Sequence[int] | NDArray[numpy.integer[Any]]`.
///
/// A `uint32` numpy array is read in place, no copy.
/// Any other sequence is copied into a `Vec<u32>` one element at a time.
pub enum TokenIds<'py> {
    Array(PyReadonlyArray1<'py, u32>),
    Copied(Vec<u32>),
}

impl<'py> FromPyObject<'_, 'py> for TokenIds<'py> {
    type Error = PyErr;

    const INPUT_TYPE: PyStaticExpr = type_hint_union!(
        type_hint_subscript!(
            type_hint_identifier!("collections.abc", "Sequence"),
            type_hint_identifier!("builtins", "int")
        ),
        type_hint_subscript!(
            type_hint_identifier!("numpy.typing", "NDArray"),
            type_hint_subscript!(
                type_hint_identifier!("numpy", "integer"),
                type_hint_identifier!("typing", "Any")
            )
        )
    );

    fn extract(obj: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(array) = obj.extract::<PyReadonlyArray1<u32>>() {
            return Ok(match array.as_slice() {
                Ok(_) => Self::Array(array),
                Err(_) => Self::Copied(array.as_array().to_vec()),
            });
        }
        obj.extract::<Vec<u32>>().map(Self::Copied)
    }
}

impl TokenIds<'_> {
    pub fn as_slice(&self) -> &[u32] {
        match self {
            Self::Array(array) => array.as_slice().expect("contiguous, checked in extract"),
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
