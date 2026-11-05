//! A utility for the macros: it turns a field type, such as `Bool`, into a Rust type.
//!
//! In `tk::schemas`, each field of a component definition has a field type:
//!
//! ```text
//! Strip { strip_left: Bool = true, strip_right: Bool = true },
//! ```
//!
//! [`component_module!`](macro@super::component_module) needs a Rust type for `Bool`: `bool`.
//! This module defines one type alias per field type, such as `arg::Bool = bool`.
//! The macro writes `arg::$field_type`, which becomes `arg::Bool`, which is `bool`:
//!
//! ```rust,ignore
//! fn new(strip_left: arg::Bool, strip_right: arg::Bool) -> ...
//!
//! #[getter]
//! fn strip_left(slf: &Bound<'_, Self>) -> PyResult<value::Bool> { ... }
//!
//! #[setter]
//! fn set_strip_left(slf: &Bound<'_, Self>, strip_left: arg::Bool) -> PyResult<()> { ... }
//! ```
//!
//! # Two Rust types per field type
//!
//! [`arg`] holds the type the constructor and the setter take.
//! [`value`] holds the type the getter returns.
//!
//! | Field type | `arg` | `value` | JSON |
//! |---|---|---|---|
//! | `Bool` | `bool` | `bool` | a bool |
//! | `OptionalBool` | `Option<bool>` | `Option<bool>` | a bool or `null` |
//! | `Usize` | `usize` | `usize` | a non-negative integer |
//! | `String` | `&str` | `String` | a string |
//! | `Char` | `char` | `char` | a string of one character |
//! | `Choice(choices)` | `&str` | `&'static str` | the `json` of one of `choices`, a list of [`Choice`] |
//! | `Pattern` | `Pattern` | `Pattern` | `{"String": "..."}` or `{"Regex": "..."}` |
//! | `Base64Bytes` | [`Base64Bytes`] | [`Base64Bytes`] | bytes, written as a base64 string |
//!
//! # Choices
//!
//! A `Choice` field holds one string from a fixed list, such as `behavior` in `Split`.
//! Python and the JSON spell each choice differently:
//! For example, Python takes `"merged_with_previous"`, and the JSON holds `"MergedWithPrevious"`.
//! So `arg::Choice` and `value::Choice` hold the Python name.
//! The constructor and the setter convert the name with [`choice_json`] before they write it.
//! The getter converts the JSON string back with [`choice_name`].
//! A name that isn't in the list raises `ValueError`.

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

/// The Rust type the constructor and the setter take, for each field type.
///
/// `String` and `Choice` are `&str`, because a default such as `"</w>"` or `"isolated"` is a
/// `&str`.
/// pyo3 needs a default to have the argument's type.
/// `"</w>".to_owned()` would also work, but pyo3 would show that default as `...` in the stubs.
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

/// The Rust type the getter returns, for each field type.
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

/// Converts a choice from its Python name to its JSON string, such as `"merged_with_previous"` to
/// `"MergedWithPrevious"`.
/// Fails if `name` is not one of `choices`. The message names the `key` field.
pub(crate) fn choice_json(key: &str, name: &str, choices: &[Choice]) -> PyResult<&'static str> {
    if let Some(choice) = choices.iter().find(|choice| choice.name == name) {
        return Ok(choice.json);
    }
    let names: Vec<&str> = choices.iter().map(|choice| choice.name).collect();
    Err(PyValueError::new_err(format!(
        "`{key}` must be one of {names:?}, got {name:?}"
    )))
}

/// Converts a choice from its JSON string to its Python name, such as `"MergedWithPrevious"` to
/// `"merged_with_previous"`.
/// Fails if `json` is not one of `choices`. The message names the `key` field.
pub(crate) fn choice_name(key: &str, json: String, choices: &[Choice]) -> PyResult<&'static str> {
    if let Some(choice) = choices.iter().find(|choice| choice.json == json) {
        return Ok(choice.name);
    }
    Err(err(format!(
        "the `{key}` field holds {json:?}, which is not a choice it takes"
    )))
}

/// Python `bytes`, stored in the JSON as a base64 string.
/// `Precompiled` stores its charsmap this way.
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
