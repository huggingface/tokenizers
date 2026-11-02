use std::sync::Arc;

use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;
use serde_json::json;

use crate::components::component_spec::{ComponentSpec, SequenceSpec};
use crate::error::err;
use crate::tokenizer::TokenizerSpec;

use super::{Component, ComponentKind, JsonFieldsExt};

const SEQUENCE_ITEMS_KEY: &str = "normalizers";

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
    const ITEMS_KEY: Option<&'static str> = Some(SEQUENCE_ITEMS_KEY);

    fn tokenizer_field_mut(spec: &mut TokenizerSpec) -> &mut Option<ComponentSpec> {
        &mut spec.normalizer
    }

    fn from_spec(py: Python<'_>, spec: ComponentSpec) -> PyResult<Py<Self>> {
        match &spec {
            ComponentSpec::Sequence(_) => Self::concrete_component(py, spec, Sequence),
            ComponentSpec::Single(single) => match single.json_type().as_deref() {
                Some("Prepend") => Self::concrete_component(py, spec, Prepend),
                _ => Self::default_component(py, spec),
            },
        }
    }
}

/// Adds `prepend` at the start of the text.
#[pyclass(frozen, extends = Normalizer, module = "tokenizers.normalizers")]
pub struct Prepend;

#[pymethods]
impl Prepend {
    #[new]
    fn new(prepend: String) -> PyClassInitializer<Self> {
        Normalizer::initializer(
            ComponentSpec::from_literal(json!({"type": "Prepend", "prepend": prepend})),
            Self,
        )
    }

    #[getter]
    fn prepend(slf: &Bound<'_, Self>) -> PyResult<String> {
        slf.field("prepend")
    }

    #[setter]
    fn set_prepend(slf: &Bound<'_, Self>, prepend: String) {
        slf.set_field("prepend", prepend);
    }
}

/// Applies each normalizer in turn.
#[pyclass(frozen, extends = Normalizer, module = "tokenizers.normalizers")]
pub struct Sequence;

#[pymethods]
impl Sequence {
    #[new]
    fn new(normalizers: Vec<Bound<'_, Normalizer>>) -> PyClassInitializer<Self> {
        let mut items = Vec::with_capacity(normalizers.len());
        for normalizer in &normalizers {
            match Normalizer::spec(normalizer) {
                ComponentSpec::Single(single) => items.push(single.clone()),
                ComponentSpec::Sequence(sequence) => items.extend(sequence.items()),
            }
        }
        let sequence = SequenceSpec::new(SEQUENCE_ITEMS_KEY, items);
        let spec = ComponentSpec::Sequence(Arc::new(sequence));
        Normalizer::initializer(spec, Self)
    }

    fn __len__(slf: &Bound<'_, Self>) -> usize {
        Normalizer::sequence_spec(slf).len()
    }

    fn __getitem__(slf: &Bound<'_, Self>, index: isize) -> PyResult<Py<Normalizer>> {
        let spec = Normalizer::sequence_spec(slf);
        let item = spec.index(index).and_then(|index| spec.get(index));
        let item = item.ok_or_else(|| PyIndexError::new_err("Sequence index out of range"))?;
        Normalizer::from_spec(slf.py(), item)
    }

    fn __setitem__(
        slf: &Bound<'_, Self>,
        index: isize,
        item: Bound<'_, Normalizer>,
    ) -> PyResult<()> {
        let spec = Normalizer::sequence_spec(slf);
        let Some(item) = Normalizer::spec(&item).as_single() else {
            return Err(err("a Sequence item can't be a Sequence"));
        };
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
