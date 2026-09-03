//! numpy arrays in and out of the module.
//!
//! rust-numpy's types carry no pyo3 introspection, so a bare [`PyArray1`] in a signature would
//! type as `Incomplete` in the generated stub. The two wrappers here give the stub real types.

use std::convert::Infallible;

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::inspect::PyStaticExpr;
use pyo3::prelude::*;
use pyo3::{Borrowed, type_hint_identifier, type_hint_subscript, type_hint_union};

/// A returned [`PyArray1<u32>`], typed `numpy.typing.NDArray[numpy.uint32]`.
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

/// The ids handed to `Tokenizer.decode`, typed `Sequence[int] | NDArray[numpy.integer[Any]]`.
///
/// A contiguous `uint32` array is read in place. Anything else, a list or an array of another
/// integer dtype, is copied into a `Vec<u32>` one element at a time.
pub enum Ids<'py> {
    Array(PyReadonlyArray1<'py, u32>),
    Copied(Vec<u32>),
}

impl<'py> FromPyObject<'_, 'py> for Ids<'py> {
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

impl Ids<'_> {
    pub fn as_slice(&self) -> &[u32] {
        match self {
            Self::Array(array) => array.as_slice().expect("contiguous, checked in extract"),
            Self::Copied(ids) => ids,
        }
    }
}
