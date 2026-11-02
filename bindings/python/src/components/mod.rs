//! Component pyclasses: the Python objects behind `tokenizer.model`, `tokenizer.normalizer`, ...
//!
//! # handle vs copy
//!
//! A tokenizer stores each component as its `tokenizer.json` object: a [`ComponentSpec`].
//! The [`Component`] `pyclass` is a handle to a shared `ComponentSpec`, which supports
//! convenient python write-through syntax:
//!
//! ```python
//! # Component aliasing
//! b.normalizer = a.normalizer
//! a.normalizer.prepend = "_"  # b.normalizer changes too
//!
//! # Mutate the tokenizer spec in place
//! tokenizer.model.ignore_merges = True
//! ```
//!
//! This design allows drop-in compatibility with tokenizers 0.23 and a more pythonic api.
//!
//! To get an independent copy of a component, use `copy.deepcopy`:
//! ```python
//! independent = copy.deepcopy(tokenizer.normalizer)
//! ```
//!
//! # Python class hierarchy
//!
//! Component (base class for all components)
//!  ⤷ Model | Normalizer | ... (component kind)
//!    ⤷ BPE | ... (concrete component)

pub(crate) mod component_spec;
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

use crate::components::component_spec::{ComponentKey, ComponentSpec};
use crate::error::err;
use crate::repr;

/// Base class for tokenizer components: models, normalizers, pre-tokenizers, post-processors
/// and decoders.
///
/// Components are handles to the underlying JSON spec. An edit to the spec touches all handles
/// pointing to the same spec:
/// ```python
/// b.normalizer = a.normalizer
/// a.normalizer.prepend = "_"  # b.normalizer changes too
/// ```
///
/// Use `copy.deepcopy(component)` for an independent copy.
///
/// Edits after `compile()` don't affect the compiled tokenizer.
#[pyclass(frozen, subclass, module = "tokenizers")]
pub struct Component {
    spec: ComponentSpec,
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
    fn from_spec(py: Python<'_>, spec: ComponentSpec) -> PyResult<Py<Self>>;

    /// A plain instance of this kind.
    fn default_component(py: Python<'_>, spec: ComponentSpec) -> PyResult<Py<Self>> {
        Py::new(py, Self::default_initializer(spec))
    }

    /// The initializer of this kind over `spec`, with no concrete class.
    fn default_initializer(spec: ComponentSpec) -> PyClassInitializer<Self> {
        PyClassInitializer::from(Component { spec }).add_subclass(Self::default())
    }

    /// Get a reference to the underlying [`ComponentSpec`]
    fn spec<'a>(handle: &'a Bound<'_, Self>) -> &'a ComponentSpec {
        &handle.as_super().get().spec
    }

    /// Builds a new instance of [`Self`] from the output of __getstate__
    /// See [`Component::__getstate__`]
    fn from_state(py: Python<'_>, state: &[u8]) -> PyResult<Py<Self>> {
        let Value::Object(json) = serde_json::from_slice(state).map_err(err)? else {
            return Err(err("a component's state must be a JSON object"));
        };
        Self::from_spec(
            py,
            ComponentSpec::from_json(Self::JSON_KEY, json).map_err(err)?,
        )
    }
}

/// Boilerplate methods for the concrete classes of a component kind: BPE, BertNormalizer, ...
///
/// Each class holds the typed spec it was built over, next to the [`ComponentSpec`] its
/// [`Component`] base holds. Both point at the same spec.
pub(crate) trait ComponentType:
    PyClass<BaseType: ComponentKind> + From<Arc<<Self as ComponentType>::Spec>>
where
    ComponentSpec: From<Arc<<Self as ComponentType>::Spec>>,
{
    /// [`SingleSpec`](component_spec::SingleSpec), or
    /// [`SequenceSpec`](component_spec::SequenceSpec) for the `Sequence` classes.
    type Spec;

    fn initializer(spec: Arc<Self::Spec>) -> PyClassInitializer<Self> {
        let base = Self::BaseType::default_initializer(ComponentSpec::from(spec.clone()));
        base.add_subclass(Self::from(spec))
    }

    /// Builds the Python object for `spec`, typed as its kind.
    fn from_spec(py: Python<'_>, spec: Arc<Self::Spec>) -> PyResult<Py<Self::BaseType>> {
        Ok(Bound::new(py, Self::initializer(spec))?
            .into_super()
            .unbind())
    }
}
