use std::sync::Arc;

use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;
use serde_json::{Value, json};

use crate::components::component_spec::{ComponentKey, ComponentSpec, SequenceSpec, SingleSpec};

use super::{Component, ComponentKind, ComponentType};

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
    const JSON_KEY: ComponentKey = ComponentKey::Normalizer;

    fn from_spec(py: Python<'_>, spec: ComponentSpec) -> PyResult<Py<Self>> {
        match &spec {
            ComponentSpec::Sequence(sequence) => Sequence::from_spec(py, sequence.clone()),
            ComponentSpec::Single(single) => match single.json_type().as_deref() {
                Some("Prepend") => Prepend::from_spec(py, single.clone()),
                _ => Self::default_component(py, spec),
            },
        }
    }
}

/// Adds `prepend` at the start of the text.
#[pyclass(frozen, extends = Normalizer, module = "tokenizers.normalizers")]
pub struct Prepend {
    spec: Arc<SingleSpec>,
}

impl From<Arc<SingleSpec>> for Prepend {
    fn from(spec: Arc<SingleSpec>) -> Self {
        Self { spec }
    }
}

impl ComponentType for Prepend {
    type Spec = SingleSpec;
}

#[pymethods]
impl Prepend {
    #[new]
    fn new(prepend: String) -> PyClassInitializer<Self> {
        let Value::Object(json) = json!({"type": "Prepend", "prepend": prepend}) else {
            unreachable!("json! builds an object from an object literal")
        };
        Self::initializer(Arc::new(SingleSpec::new(json)))
    }

    #[getter]
    fn prepend(&self) -> PyResult<String> {
        self.spec.field("prepend")
    }

    #[setter]
    fn set_prepend(&self, prepend: String) {
        self.spec.set_field("prepend", prepend);
    }
}

/// Applies each normalizer in turn.
#[pyclass(frozen, extends = Normalizer, module = "tokenizers.normalizers")]
pub struct Sequence {
    spec: Arc<SequenceSpec>,
}

impl From<Arc<SequenceSpec>> for Sequence {
    fn from(spec: Arc<SequenceSpec>) -> Self {
        Self { spec }
    }
}

impl ComponentType for Sequence {
    type Spec = SequenceSpec;
}

#[pymethods]
impl Sequence {
    #[new]
    fn new(normalizers: Vec<Bound<'_, Normalizer>>) -> PyResult<PyClassInitializer<Self>> {
        let items: Vec<ComponentSpec> = normalizers
            .iter()
            .map(|item| Normalizer::spec(item).clone())
            .collect();
        let sequence = SequenceSpec::new(Normalizer::JSON_KEY, items)?;
        Ok(Self::initializer(Arc::new(sequence)))
    }

    fn __len__(&self) -> usize {
        self.spec.len()
    }

    fn __getitem__(slf: &Bound<'_, Self>, index: isize) -> PyResult<Py<Normalizer>> {
        let spec = &slf.get().spec;
        let item = spec.index(index).and_then(|index| spec.get(index));
        let item = item.ok_or_else(|| PyIndexError::new_err("Sequence index out of range"))?;
        Normalizer::from_spec(slf.py(), item)
    }

    fn __setitem__(&self, index: isize, item: Bound<'_, Normalizer>) -> PyResult<()> {
        let item = Arc::try_from(Normalizer::spec(&item).clone())?;
        self.spec
            .index(index)
            .and_then(|index| self.spec.set(index, item))
            .ok_or_else(|| PyIndexError::new_err("Sequence assignment index out of range"))
    }
}

#[pymodule]
pub mod normalizers {
    #[pymodule_export]
    use super::{Normalizer, Prepend, Sequence};
}
