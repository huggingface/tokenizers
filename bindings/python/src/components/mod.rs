//! Component pyclasses to support write through efficiently:
//!
//! ```python
//! tokenizer.pre_tokenizer = NFC()
//! tokenizer.model.ignore_merges = True
//! ```
//!
//! Python class hierarchy:
//!
//! Component (base class for all components)
//!  ⤷ Model | Normalizer | ... (component kind)
//!    ⤷ BPE | ... (concrete component)

pub(crate) mod decoders;
pub(crate) mod models;
pub(crate) mod normalizers;
pub(crate) mod pre_tokenizers;
pub(crate) mod processors;

use std::sync::Arc;

use pyo3::PyClass;
use pyo3::impl_::pyclass::PyClassBaseType;
use pyo3::prelude::*;
use serde_json::Value;

use crate::error::err;
use crate::repr;
use crate::spec::{ComponentKey, ComponentSpec};

/// Base class for tokenizer components
#[pyclass(frozen, subclass, module = "tokenizers")]
pub struct Component {
    spec: Arc<ComponentSpec>,
}

#[pymethods]
impl Component {
    fn __repr__(&self) -> String {
        repr::component(&self.spec.json())
    }

    /// The component's JSON, as `bytes`.
    fn __getstate__(&self) -> Vec<u8> {
        serde_json::to_vec(&*self.spec.json()).expect("a JSON object always serializes")
    }

    /// Pickle rebuilds a component through its kind's `_unpickle`.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, (Vec<u8>,))> {
        Ok((
            slf.get_type().getattr("_unpickle")?,
            (slf.get().__getstate__(),),
        ))
    }
}

/// Boilerplate implementation for subclasses of Component representing component kinds:
/// Model, Normalizer, PreTokenizer, ...
pub(crate) trait ComponentKind:
    PyClass<BaseType = Component> + PyClassBaseType<Initializer = PyClassInitializer<Self>> + Default
{
    /// The `tokenizer.json` key where the spec for this component kind is stored.
    /// For example, `"pre_tokenizer"` for the `PreTokenizer` component.
    const JSON_KEY: ComponentKey;

    /// Override this to instantiate leaf classes (eg, instantiate a BPE for models)
    fn from_spec(py: Python<'_>, spec: Arc<ComponentSpec>) -> PyResult<Py<Self>> {
        Py::new(py, Self::make_init(spec))
    }

    fn make_init(spec: Arc<ComponentSpec>) -> PyClassInitializer<Self> {
        PyClassInitializer::from(Component { spec }).add_subclass(Self::default())
    }

    /// Builds `subclass` around `spec`, for example a `BPE` for a model whose `"type"` is `"BPE"`.
    fn subclass_from_spec<T: PyClass<BaseType = Self>>(
        py: Python<'_>,
        spec: Arc<ComponentSpec>,
        subclass: T,
    ) -> PyResult<Py<Self>> {
        let init = Self::make_init(spec).add_subclass(subclass);
        Ok(Bound::new(py, init)?.into_super().unbind())
    }

    /// Get a reference to the underlying [`ComponentSpec`]
    fn spec<'a>(handle: &'a Bound<'_, Self>) -> &'a Arc<ComponentSpec> {
        &handle.as_super().get().spec
    }

    /// Builds a new instance of [`Self`] from the output of __getstate__
    /// See [`ComponentHandle::__getstate__`]
    fn from_state(py: Python<'_>, state: &[u8]) -> PyResult<Py<Self>> {
        let Value::Object(json) = serde_json::from_slice(state).map_err(err)? else {
            return Err(err("a component's state must be a JSON object"));
        };
        Self::from_spec(py, ComponentSpec::new(json))
    }
}

/// Reads and writes one field of a component's JSON.
pub(crate) trait JsonFields {
    /// A missing key reads as `null`, so `Option<T>` reads it as `None`.
    fn field<T: serde::de::DeserializeOwned>(&self, key: &str) -> PyResult<T>;
    fn set_field(&self, key: &str, value: impl Into<Value>);
}

impl<K: ComponentKind, T: PyClass<BaseType = K>> JsonFields for Bound<'_, T> {
    fn field<V: serde::de::DeserializeOwned>(&self, key: &str) -> PyResult<V> {
        let json = K::spec(self.as_super()).json();
        let value = json.get(key).unwrap_or(&Value::Null);
        V::deserialize(value).map_err(|e| err(format!("`{key}`: {e}")))
    }

    fn set_field(&self, key: &str, value: impl Into<Value>) {
        K::spec(self.as_super()).edit(|json| json.insert(key.to_owned(), value.into()));
    }
}
