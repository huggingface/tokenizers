//! Component pyclasses:
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
use pyo3::exceptions::PyIndexError;
use pyo3::impl_::pyclass::PyClassBaseType;
use pyo3::prelude::*;
use serde_json::{Value, json};

use crate::JsonObject;
use crate::error::err;
use crate::repr;
use crate::tokenizer::{ComponentKey, ComponentSpec};

/// Base class for stand-alone / detached tokenizer components
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

/// Boilerplate methods for subclasses of Component representing component kinds:
/// Model, Normalizer, PreTokenizer, ...
pub(crate) trait ComponentKind:
    PyClass<BaseType = Component> + PyClassBaseType<Initializer = PyClassInitializer<Self>> + Default
{
    /// The `tokenizer.json` key where the spec for this component kind is stored.
    /// For example, `ComponentKey::PreTokenizer` for the `PreTokenizer` component.
    const JSON_KEY: ComponentKey;

    /// Builds the Python object for `spec`.
    fn from_spec(py: Python<'_>, spec: Arc<ComponentSpec>) -> PyResult<Py<Self>>;

    /// A plain instance of this kind.
    fn default_component(py: Python<'_>, spec: Arc<ComponentSpec>) -> PyResult<Py<Self>> {
        Py::new(py, Self::default_initializer(spec))
    }

    /// The initializer of this kind over `spec`, with no concrete class.
    fn default_initializer(spec: Arc<ComponentSpec>) -> PyClassInitializer<Self> {
        PyClassInitializer::from(Component { spec }).add_subclass(Self::default())
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

/// Boilerplate methods for the concrete classes of a component kind: BPE, BertNormalizer, ...
pub(crate) trait ComponentType: PyClass<BaseType: ComponentKind> + Default {
    fn initializer(spec: Arc<ComponentSpec>) -> PyClassInitializer<Self> {
        Self::BaseType::default_initializer(spec).add_subclass(Self::default())
    }

    /// Builds the Python object for `spec`, typed as its kind.
    fn from_spec(py: Python<'_>, spec: Arc<ComponentSpec>) -> PyResult<Py<Self::BaseType>> {
        Ok(Bound::new(py, Self::initializer(spec))?
            .into_super()
            .unbind())
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

/// The `Sequence` of a component kind: its JSON lists components of that kind under `ITEMS_KEY`.
///
/// The JSON holds copies: building a sequence copies its items, and reading an item gives a copy.
/// `seq[i] = item` writes into the sequence, and so into a tokenizer holding it.
#[expect(dead_code, reason = "the component kinds use it")]
pub(crate) trait ComponentSequence: PyClass<BaseType = Self::Kind> {
    type Kind: ComponentKind;

    /// The JSON key of the list, for example `"pretokenizers"` for `pre_tokenizers.Sequence`.
    const ITEMS_KEY: &'static str;

    fn from_items(items: Vec<Bound<'_, Self::Kind>>, sequence: Self) -> PyClassInitializer<Self> {
        let items: Vec<Value> = items.iter().map(item_json).collect();
        let Value::Object(json) = json!({"type": "Sequence", Self::ITEMS_KEY: items}) else {
            unreachable!("json! builds an object from an object literal")
        };
        Self::Kind::default_initializer(ComponentSpec::new(json)).add_subclass(sequence)
    }

    fn len(slf: &Bound<'_, Self>) -> usize {
        let json = Self::Kind::spec(slf.as_super()).json();
        json[Self::ITEMS_KEY].as_array().map_or(0, Vec::len)
    }

    fn get(slf: &Bound<'_, Self>, index: usize) -> PyResult<Py<Self::Kind>> {
        let json = Self::Kind::spec(slf.as_super()).json();
        let Some(Value::Object(item)) = json[Self::ITEMS_KEY].get(index) else {
            return Err(PyIndexError::new_err("Index not found"));
        };
        Self::Kind::from_spec(slf.py(), ComponentSpec::new(item.clone()))
    }

    fn set(slf: &Bound<'_, Self>, index: usize, item: &Bound<'_, Self::Kind>) -> PyResult<()> {
        let item = item_json(item);
        Self::Kind::spec(slf.as_super()).edit(|json| {
            let slot = json
                .get_mut(Self::ITEMS_KEY)
                .and_then(|items| items.get_mut(index));
            let slot = slot.ok_or_else(|| PyIndexError::new_err("Index not found"))?;
            *slot = item;
            Ok(())
        })
    }
}

fn item_json<K: ComponentKind>(item: &Bound<'_, K>) -> Value {
    Value::Object(JsonObject::clone(&K::spec(item).json()))
}
