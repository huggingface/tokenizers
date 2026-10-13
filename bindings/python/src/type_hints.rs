//! Argument and return types whose type hints pyo3 cannot derive

use std::convert::Infallible;

use numpy::{
    PyArray1, PyArrayDescrMethods, PyReadonlyArray1, PyUntypedArray, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyTypeError;
use pyo3::inspect::PyStaticExpr;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyString};
use pyo3::{Borrowed, PyTypeInfo, type_hint_identifier, type_hint_subscript, type_hint_union};

use crate::encoding::Encoding;

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

/// The `ids` argument of `Tokenizer.decode` and `Tokenizer.decode_tokens`, typed
/// `Encoding | Sequence[int] | NDArray[numpy.integer[Any]]`.
///
/// An `Encoding` and a `uint32` numpy array are read in place, no copy.
/// Contiguous native `int32` and `int64` arrays are range-checked and
/// copied in Rust. Other sequences use Python's integer conversion.
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
        // Preserve custom iteration on ndarray subclasses.
        if obj.is_exact_instance_of::<PyUntypedArray>() {
            return Self::extract_numpy_ids(obj).map(Self::Copied);
        }
        obj.extract::<Vec<u32>>().map(Self::Copied)
    }
}

impl TokenIds<'_> {
    // Avoid increasing code size in the Encoding and uint32 fast paths.
    #[inline(never)]
    fn extract_numpy_ids(obj: Borrowed<'_, '_, PyAny>) -> PyResult<Vec<u32>> {
        let len = match obj.len() {
            Ok(len) => len,
            Err(_) => return obj.extract::<Vec<u32>>(),
        };

        if len == 0 {
            return Ok(Vec::new());
        }

        if let Some(ids) = Self::copy_numpy_integer_ids(obj) {
            return Ok(ids);
        }

        let mut ids = Vec::with_capacity(len);
        for item in obj.try_iter()? {
            ids.push(item?.extract::<u32>()?);
        }

        Ok(ids)
    }

    #[inline(never)]
    fn copy_numpy_integer_ids(obj: Borrowed<'_, '_, PyAny>) -> Option<Vec<u32>> {
        let array = obj.cast::<PyUntypedArray>().ok()?;
        if array.ndim() != 1 || !array.is_contiguous() || !array.is_aligned() {
            return None;
        }

        let dtype = array.dtype();
        if !matches!((dtype.kind(), dtype.itemsize()), (b'i', 4 | 8))
            || dtype.is_native_byteorder() != Some(true)
        {
            return None;
        }

        // Preserve Python's overflow errors through scalar conversion.
        macro_rules! copy_integer_array {
            ($ty:ty) => {
                if let Ok(array) = obj.extract::<PyReadonlyArray1<$ty>>()
                    && let Ok(slice) = array.as_slice()
                    && let Ok(ids) = slice
                        .iter()
                        .map(|&id| u32::try_from(id))
                        .collect::<Result<Vec<_>, _>>()
                {
                    return Some(ids);
                }
            };
        }

        copy_integer_array!(i32);
        copy_integer_array!(i64);

        None
    }

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
