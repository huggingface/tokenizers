use std::sync::Arc;

use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;
use serde_json::{Value, json};

use crate::components::component_spec::ComponentSpec;
use crate::error::err;

use super::{Component, ComponentKind, JsonFields};

/// Base class for all normalizers
#[pyclass(frozen, subclass, extends = Component, module = "tokenizers.normalizers")]
#[derive(Default)]
pub struct Normalizer;

#[pymethods]
impl Normalizer {
    /// Rebuilds a component from its `__getstate__`
    #[staticmethod]
    fn _unpickle(py: Python<'_>, state: &[u8]) -> PyResult<Py<Self>> {
        Self::from_state(py, state)
    }
}

impl ComponentKind for Normalizer {
    const JSON_KEY: &'static str = "normalizer";
    const ITEMS_KEY: Option<&'static str> = Some("normalizers");

    fn from_spec(py: Python<'_>, spec: Arc<ComponentSpec>) -> PyResult<Py<Self>> {
        if spec.items_key().is_some() {
            return Self::of::<Sequence>(py, spec);
        }
        match spec.json_type().as_deref() {
            Some("Prepend") => Self::of::<Prepend>(py, spec),
            _ => Self::default_component(py, spec),
        }
    }
}

/// Adds `prepend` at the start of the text.
#[pyclass(frozen, extends = Normalizer, module = "tokenizers.normalizers")]
#[derive(Default)]
pub struct Prepend;

#[pymethods]
impl Prepend {
    #[new]
    fn new(prepend: String) -> PyClassInitializer<Self> {
        let Value::Object(json) = json!({"type": "Prepend", "prepend": prepend}) else {
            unreachable!("json! builds an object from an object literal")
        };
        Normalizer::default_initializer(ComponentSpec::new(json)).add_subclass(Self)
    }

    #[getter]
    fn prepend(slf: &Bound<'_, Self>) -> PyResult<String> {
        slf.field("prepend")
    }

    #[setter]
    fn set_prepend(slf: &Bound<'_, Self>, prepend: String) {
        slf.set_field("prepend", prepend)
    }
}

/// Applies each normalizer in turn.
#[pyclass(frozen, extends = Normalizer, module = "tokenizers.normalizers")]
#[derive(Default)]
pub struct Sequence;

#[pymethods]
impl Sequence {
    #[new]
    fn new(normalizers: Vec<Bound<'_, Normalizer>>) -> PyResult<PyClassInitializer<Self>> {
        let items: Vec<Arc<ComponentSpec>> = normalizers
            .iter()
            .map(|item| Normalizer::spec(item).clone())
            .collect();
        let items_key = Normalizer::ITEMS_KEY.expect("Normalizer lists its Sequence items");
        Ok(
            Normalizer::default_initializer(ComponentSpec::sequence(items_key, items))
                .add_subclass(Self),
        )
    }

    fn __len__(slf: &Bound<'_, Self>) -> usize {
        Normalizer::spec(slf.as_super()).len()
    }

    fn __getitem__(slf: &Bound<'_, Self>, index: isize) -> PyResult<Py<Normalizer>> {
        let spec = Normalizer::spec(slf.as_super());
        let item = spec.index(index).and_then(|index| spec.get(index));
        let item = item.ok_or_else(|| PyIndexError::new_err("Sequence index out of range"))?;
        Normalizer::from_spec(slf.py(), item)
    }

    fn __setitem__(
        slf: &Bound<'_, Self>,
        index: isize,
        item: Bound<'_, Normalizer>,
    ) -> PyResult<()> {
        let spec = Normalizer::spec(slf.as_super());
        let item = Normalizer::spec(&item);
        if Arc::ptr_eq(spec, item) {
            return Err(err(
                "a Sequence can't hold itself: its JSON is rebuilt from its items",
            ));
        }
        spec.index(index)
            .and_then(|index| spec.set(index, item.clone()))
            .ok_or_else(|| PyIndexError::new_err("Sequence assignment index out of range"))
    }
}

#[pymodule]
pub mod normalizers {
    #[pymodule_export]
    use super::{Normalizer, Prepend, Sequence};
}
