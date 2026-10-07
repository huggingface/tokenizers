//! Argument and return types whose type hints pyo3 cannot derive

use std::convert::Infallible;

use numpy::{Element, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyTypeError;
use pyo3::inspect::PyStaticExpr;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyString};
use pyo3::{Borrowed, PyTypeInfo, type_hint_identifier, type_hint_subscript, type_hint_union};

use crate::encoding::Encoding;

/// A 1-D numpy array, typed `NDArray[numpy.<dtype>]`.
///
/// rust-numpy has no pyo3 introspection, so a bare [`PyArray1`] in a signature is typed `Incomplete`.
pub struct NDArray<'py, T: Dtype>(pub Bound<'py, PyArray1<T>>);

pub type U32Array<'py> = NDArray<'py, u32>;

/// An element type of [`NDArray`], with the numpy scalar type that names it in the type hints.
pub trait Dtype: Element {
    const TYPE_HINT: PyStaticExpr;
}

impl Dtype for u32 {
    const TYPE_HINT: PyStaticExpr = type_hint_identifier!("numpy", "uint32");
}

impl<'py, T: Dtype> IntoPyObject<'py> for NDArray<'py, T> {
    type Target = PyArray1<T>;
    type Output = Bound<'py, PyArray1<T>>;
    type Error = Infallible;

    const OUTPUT_TYPE: PyStaticExpr = type_hint_subscript!(
        type_hint_identifier!("numpy.typing", "NDArray"),
        T::TYPE_HINT
    );

    fn into_pyobject(self, _py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(self.0)
    }
}

/// The `ids` argument of `Tokenizer.decode` and `Tokenizer.decode_tokens`, typed
/// `Encoding | Sequence[int] | NDArray[numpy.integer[Any]]`.
///
/// An `Encoding` and a `uint32` numpy array are read in place, no copy.
/// Any other sequence is copied into a `Vec<u32>` one element at a time.
pub enum TokenIds<'py> {
    Encoding(Bound<'py, Encoding>),
    Array(PyReadonlyArray1<'py, u32>),
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
        type_hint_subscript!(
            type_hint_identifier!("numpy.typing", "NDArray"),
            type_hint_subscript!(
                type_hint_identifier!("numpy", "integer"),
                type_hint_identifier!("typing", "Any")
            )
        )
    );

    fn extract(obj: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(encoding) = obj.extract::<Bound<'py, Encoding>>() {
            return Ok(Self::Encoding(encoding));
        }
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
            Self::Encoding(encoding) => encoding.get().ids(),
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
