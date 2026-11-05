//! The classes of the simple components, written by [`py_classes!`] from the core's lists.
//!
//! `tk::normalizer_schemas!(py_classes! { ... })` hands each normalizer's entry (its doc, its JSON
//! `"type"` and its fields) to [`py_classes!`], which writes, for the kind:
//!
//! 1. the kind's `Sequence` class,
//! 2. one class per component, with a constructor, and a getter and a setter per field,
//! 3. `from_json_type`, which picks the class of a component from its `"type"`,
//! 4. the kind's `#[pymodule]`.
//!
//! A field's shape, such as `Bool`, names a type in [`arg`] (what the constructor and the setter
//! take) and in [`value`] (what the getter returns). Each type moves to and from the JSON with
//! serde, so a new field shape is one type here, and the macro does not change.

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

/// Writes a kind's classes from its `tk::*_schemas!` entries. The arguments are the kind's class,
/// its `#[pymodule]`, the Python module of its classes and the doc of its `Sequence`.
macro_rules! py_classes {
    ({ $kind:ident, $module:ident, $python_module:literal, $sequence_doc:literal } $(
        $(#[doc = $doc:literal])+
        $class:ident $({
            $($key:ident: $field_type:ident $(($choices:expr))? $(= $default:tt)?),* $(,)?
        })?
    ),* $(,)?) => {
        // pyo3 writes the `__setitem__` slot as an item next to the impl, calling an unsafe fn
        // through `Sequence`, whose span is this macro's. The allow has to sit on a module.
        #[allow(unsafe_op_in_unsafe_fn)]
        mod sequence {
            use super::*;

            #[doc = $sequence_doc]
            #[pyclass(frozen, extends = $kind, module = $python_module)]
            pub struct Sequence;

            #[pymethods]
            impl Sequence {
                #[new]
                #[pyo3(signature = (items, /))]
                fn new(items: Vec<Bound<'_, $kind>>) -> PyClassInitializer<Self> {
                    use crate::components::component_spec::ComponentSpec;
                    use crate::components::ComponentKind;

                    let mut flat = Vec::with_capacity(items.len());
                    for item in &items {
                        match $kind::spec(item) {
                            ComponentSpec::Single(single) => flat.push(single.clone()),
                            ComponentSpec::Sequence(sequence) => flat.extend(sequence.items()),
                        }
                    }
                    let items_key = $kind::ITEMS_KEY.expect("a kind with a Sequence has an items key");
                    let sequence = crate::components::component_spec::SequenceSpec::new(items_key, flat);
                    $kind::initializer(ComponentSpec::Sequence(std::sync::Arc::new(sequence)), Self)
                }

                fn __len__(slf: &Bound<'_, Self>) -> usize {
                    <$kind as crate::components::ComponentKind>::sequence_spec(slf).len()
                }

                fn __getitem__(slf: &Bound<'_, Self>, index: isize) -> PyResult<Py<$kind>> {
                    use crate::components::ComponentKind;

                    let spec = $kind::sequence_spec(slf);
                    let item = spec.index(index).and_then(|index| spec.get(index));
                    let item = item.ok_or_else(|| {
                        pyo3::exceptions::PyIndexError::new_err("Sequence index out of range")
                    })?;
                    $kind::from_spec(slf.py(), item)
                }

                fn __setitem__(
                    slf: &Bound<'_, Self>,
                    index: isize,
                    item: Bound<'_, $kind>,
                ) -> PyResult<()> {
                    use crate::components::ComponentKind;

                    let spec = $kind::sequence_spec(slf);
                    let Some(item) = $kind::spec(&item).as_single() else {
                        return Err(crate::error::err("a Sequence item can't be a Sequence"));
                    };
                    spec.index(index)
                        .and_then(|index| spec.set(index, item.clone()))
                        .ok_or_else(|| {
                            pyo3::exceptions::PyIndexError::new_err(
                                "Sequence assignment index out of range",
                            )
                        })
                }
            }
        }
        pub use sequence::Sequence;

        $(
            $(#[doc = $doc])+
            #[allow(clippy::upper_case_acronyms)]
            #[pyclass(frozen, extends = $kind, module = $python_module)]
            pub struct $class;

            // pyo3 reads the methods before any macro inside them expands, so `paste!` names the
            // setters first.
            pastey::paste! {
                #[pymethods]
                impl $class {
                    #[new]
                    #[pyo3(signature = ($($($key $(= $default)?),*)?))]
                    fn new(
                        $($($key: crate::components::fields::arg::$field_type),*)?
                    ) -> PyResult<PyClassInitializer<Self>> {
                        #[allow(unused_mut)]
                        let mut json = serde_json::Map::new();
                        json.insert("type".into(), stringify!($class).into());
                        $($(
                            $(crate::components::check_choice(stringify!($key), $key, $choices)?;)?
                            let $key = serde_json::to_value($key).map_err(crate::error::err)?;
                            json.insert(stringify!($key).into(), $key);
                        )*)?
                        let spec = crate::components::component_spec::ComponentSpec::from_literal(
                            serde_json::Value::Object(json),
                        );
                        Ok(<$kind as crate::components::ComponentKind>::initializer(spec, Self))
                    }

                    $($(
                        #[getter]
                        fn $key(
                            slf: &Bound<'_, Self>,
                        ) -> PyResult<crate::components::fields::value::$field_type> {
                            crate::components::JsonFieldsExt::field(slf, stringify!($key))
                        }

                        #[setter]
                        fn [<set_ $key>](
                            slf: &Bound<'_, Self>,
                            $key: crate::components::fields::arg::$field_type,
                        ) -> PyResult<()> {
                            $(crate::components::check_choice(stringify!($key), $key, $choices)?;)?
                            let $key = serde_json::to_value($key).map_err(crate::error::err)?;
                            crate::components::JsonFieldsExt::set_field(slf, stringify!($key), $key);
                            Ok(())
                        }
                    )*)?
                }
            }
        )*

        /// Builds a handle of the class `json_type` names, or a plain kind handle for another type.
        pub(crate) fn from_json_type(
            py: Python<'_>,
            spec: crate::components::component_spec::ComponentSpec,
            json_type: Option<&str>,
        ) -> PyResult<Py<$kind>> {
            use crate::components::ComponentKind;

            $(if json_type == Some(stringify!($class)) {
                return $kind::concrete_component(py, spec, $class);
            })*
            $kind::default_component(py, spec)
        }

        #[pymodule]
        pub mod $module {
            #[pymodule_export]
            use super::{$kind, Sequence, $($class),*};
        }
    };
}

pub(crate) use py_classes;
