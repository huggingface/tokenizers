//! The types of the field shapes in `tk_serialize::schemas`.
//!
//! A field's shape, such as `Bool`, names a type in [`arg`] (what the constructor and the setter
//! take) and in [`value`] (what the getter returns). Each type moves to and from the JSON with
//! serde, so a new field shape is one type here, and
//! [`component_classes!`](super::macros::component_classes) does not change.

use std::convert::Infallible;

use base64::Engine as _;
use pyo3::inspect::PyStaticExpr;
use pyo3::prelude::*;
use pyo3::type_object::PyTypeInfo;
use pyo3::types::PyBytes;
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// What the constructor and the setter take from Python, by field shape.
pub(crate) mod arg {
    pub(crate) type Bool = bool;
    pub(crate) type String<'a> = &'a str;
    pub(crate) type Char = char;
    pub(crate) type Usize = usize;
    pub(crate) type OptionalBool = Option<bool>;
    pub(crate) type Choice<'a> = &'a str;
    pub(crate) type Pattern = crate::regex::Pattern;
    pub(crate) type Base64Bytes = super::Base64Bytes;
}

/// What the getter returns to Python, by field shape.
pub(crate) mod value {
    pub(crate) type Bool = bool;
    pub(crate) type String = std::string::String;
    pub(crate) type Char = char;
    pub(crate) type Usize = usize;
    pub(crate) type OptionalBool = Option<bool>;
    pub(crate) type Choice = std::string::String;
    pub(crate) type Pattern = crate::regex::Pattern;
    pub(crate) type Base64Bytes = super::Base64Bytes;
}

/// Python `bytes`, written in the JSON as a base64 string, as `Precompiled`'s charsmap is.
pub(crate) struct Base64Bytes(Vec<u8>);

impl FromPyObject<'_, '_> for Base64Bytes {
    type Error = PyErr;
    const INPUT_TYPE: PyStaticExpr = PyBytes::TYPE_HINT;

    fn extract(ob: Borrowed<'_, '_, PyAny>) -> PyResult<Self> {
        Ok(Self(ob.extract::<&[u8]>()?.to_vec()))
    }
}

impl<'py> IntoPyObject<'py> for Base64Bytes {
    type Target = PyBytes;
    type Output = Bound<'py, PyBytes>;
    type Error = Infallible;
    const OUTPUT_TYPE: PyStaticExpr = PyBytes::TYPE_HINT;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(PyBytes::new(py, &self.0))
    }
}

impl Serialize for Base64Bytes {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&tk::BASE64.encode(&self.0))
    }
}

impl<'de> Deserialize<'de> for Base64Bytes {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let encoded = String::deserialize(deserializer)?;
        tk::BASE64
            .decode(encoded)
            .map(Self)
            .map_err(D::Error::custom)
    }
}
