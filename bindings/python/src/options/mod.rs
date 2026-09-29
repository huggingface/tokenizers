pub mod padding;
pub mod truncation;

pub use padding::Padding;
pub use truncation::Truncation;

use pyo3::Borrowed;
use pyo3::inspect::PyStaticExpr;
use pyo3::prelude::*;

use tk_encode::{PaddingParams, TruncationParams};

/// Sentinel type used to differentiate tok.encode(text, padding=None) from tok.encode(text)
/// (explicit None = disabled vs omitted = default)
#[derive(PartialEq, Hash)]
pub enum OverrideSentinel<T> {
    InheritConfig,
    Off,
    With(T),
}

impl FromPyObject<'_, '_> for OverrideSentinel<PaddingParams> {
    type Error = PyErr;

    const INPUT_TYPE: PyStaticExpr =
        <Option<PyRef<'static, Padding>> as FromPyObject<'static, 'static>>::INPUT_TYPE;

    fn extract(obj: Borrowed<'_, '_, PyAny>) -> Result<Self, Self::Error> {
        Ok(match obj.extract::<Option<PyRef<'_, Padding>>>()? {
            None => Self::Off,
            Some(padding) => Self::With(padding.params().clone()),
        })
    }
}

impl FromPyObject<'_, '_> for OverrideSentinel<TruncationParams> {
    type Error = PyErr;

    const INPUT_TYPE: PyStaticExpr =
        <Option<PyRef<'static, Truncation>> as FromPyObject<'static, 'static>>::INPUT_TYPE;

    fn extract(obj: Borrowed<'_, '_, PyAny>) -> Result<Self, Self::Error> {
        Ok(match obj.extract::<Option<PyRef<'_, Truncation>>>()? {
            None => Self::Off,
            Some(truncation) => Self::With(truncation.params().clone()),
        })
    }
}
