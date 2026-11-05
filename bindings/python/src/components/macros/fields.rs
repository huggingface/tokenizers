//! The types of the field shapes in `tk_serialize::schemas`.
//!
//! A field's shape, such as `Bool`, names a type in [`arg`] (what the constructor and the setter
//! take) and in [`value`] (what the getter returns). Each type moves to and from the JSON with
//! serde, so a new field shape is one type here, and
//! [`component_classes!`](super::component_classes) does not change.

use std::convert::Infallible;

use base64::Engine as _;
use pyo3::exceptions::PyValueError;
use pyo3::inspect::PyStaticExpr;
use pyo3::prelude::*;
use pyo3::type_object::PyTypeInfo;
use pyo3::types::PyBytes;
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use tk::schemas::Choice;

use crate::error::err;

/// What the constructor and the setter take from Python, by field shape.
///
/// A string is a `&str`, the type of a schema's default literal: pyo3 gives a default the
/// argument's type, and writes any other expression as `...` in the stubs.
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
    pub(crate) type Choice = &'static str;
    pub(crate) type Pattern = crate::regex::Pattern;
    pub(crate) type Base64Bytes = super::Base64Bytes;
}

/// Returns the string `tokenizer.json` holds for the choice `name` of the `key` field.
pub(crate) fn choice_json(key: &str, name: &str, choices: &[Choice]) -> PyResult<&'static str> {
    if let Some(choice) = choices.iter().find(|choice| choice.name == name) {
        return Ok(choice.json);
    }
    let names: Vec<&str> = choices.iter().map(|choice| choice.name).collect();
    Err(PyValueError::new_err(format!(
        "`{key}` must be one of {names:?}, got {name:?}"
    )))
}

/// Returns the name of the choice whose string in `tokenizer.json` is `json`.
pub(crate) fn choice_name(key: &str, json: String, choices: &[Choice]) -> PyResult<&'static str> {
    if let Some(choice) = choices.iter().find(|choice| choice.json == json) {
        return Ok(choice.name);
    }
    Err(err(format!(
        "the `{key}` field holds {json:?}, which is not a choice it takes"
    )))
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
