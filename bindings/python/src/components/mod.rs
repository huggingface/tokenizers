//! The Python classes behind `tokenizer.model`, `tokenizer.normalizer` and the other components.
//!
//! # Handles
//!
//! A tokenizer stores each component as a *spec*: the component's JSON representation in `tokenizer.json`.
//! [`TokenizerSpec`] has one field per component (`normalizer`, ..., `model`), each holding a spec.
//!
//! [`TokenizerSpec`]: crate::tokenizer::TokenizerSpec
//!
//! A Python component object is a *handle*: it holds no data of its own, only a pointer to a spec.
//! Reading `tokenizer.normalizer` returns a handle on the tokenizer's own spec, not a copy.
//! Assigning `tokenizer.normalizer = n` makes the tokenizer share `n`'s spec, again without a copy.
//!
//! So an edit through any handle reaches the tokenizer, and every other handle on the same spec:
//!
//! ```python
//! b.normalizer = a.normalizer
//! a.normalizer.prepend = "_"            # b.normalizer changes too
//! tokenizer.model.ignore_merges = True  # edits the tokenizer's model in place
//! ```
//!
//! `copy.deepcopy` returns an independent copy:
//!
//! ```python
//! independent = copy.deepcopy(tokenizer.normalizer)
//! ```
//!
//! [`component_spec`] explains how specs are stored, shared and copied.
//!
//! # Python class hierarchy
//!
//! ```text
//! Component
//!  ⤷ Model | Normalizer | ...         component kinds
//!    ⤷ Prepend | Sequence | ...       concrete classes
//! ```
//!
//! - [`Component`] holds the spec as a [`ComponentSpec`]: one component, or a sequence.
//! - A *kind*, such as [`Normalizer`](normalizers::Normalizer), holds no data. It implements
//!   [`ComponentKind`], which knows the kind's `tokenizer.json` key.
//! - A *concrete class*, such as [`Prepend`](normalizers::Prepend), implements [`ComponentType`].
//!   It keeps a typed pointer to the same spec.
//!
//! The [`Tokenizer`](crate::tokenizer::Tokenizer) getters and `Sequence.__getitem__` build a new
//! handle on each call. [`ComponentKind::from_spec`] picks the class from the spec's `"type"`.
//! A type with no class of its own gets the base kind.

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

/// Base class of every tokenizer component: models, normalizers, pre-tokenizers, post-processors
/// and decoders.
///
/// A component points at a shared spec object.
/// An edit through one component reaches every component that points at the same spec:
/// ```python
/// b.normalizer = a.normalizer
/// a.normalizer.prepend = "_"  # b.normalizer changes too
/// ```
///
/// `copy.deepcopy(component)` returns an independent copy.
///
/// A compiled tokenizer ignores later edits. Call `compile()` again to apply them.
#[pyclass(frozen, subclass, module = "tokenizers")]
pub struct Component {
    spec: ComponentSpec,
}

#[pymethods]
impl Component {
    fn __repr__(&self) -> String {
        repr::component(&self.spec.json())
    }

    /// Returns the component's JSON, as `bytes`.
    fn __getstate__(&self) -> Vec<u8> {
        serde_json::to_vec(&*self.spec.json()).expect("a JSON object always serializes")
    }

    /// Pickles the component as its JSON. Unpickling rebuilds it through its kind's `_unpickle`.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, (Vec<u8>,))> {
        Ok((
            slf.get_type().getattr("_unpickle")?,
            (slf.get().__getstate__(),),
        ))
    }
}

/// The shared methods of a component kind (for example, `Model` or `Normalizer`).
///
/// A kind holds no data. Its [`Component`] base holds the spec.
pub(crate) trait ComponentKind:
    PyClass<BaseType = Component> + PyClassBaseType<Initializer = PyClassInitializer<Self>> + Default
{
    /// The kind's top-level key in `tokenizer.json`, such as [`ComponentKey::PreTokenizer`].
    const JSON_KEY: ComponentKey;

    /// Builds a handle on `spec`, of the concrete class its `"type"` names.
    fn from_spec(py: Python<'_>, spec: ComponentSpec) -> PyResult<Py<Self>>;

    /// Builds a handle on `spec` of the plain kind, with no concrete class.
    fn default_component(py: Python<'_>, spec: ComponentSpec) -> PyResult<Py<Self>> {
        Py::new(py, Self::default_initializer(spec))
    }

    /// Returns the pyo3 initializer of the plain kind over `spec`.
    /// [`ComponentType::initializer`] adds a concrete class on top of it.
    fn default_initializer(spec: ComponentSpec) -> PyClassInitializer<Self> {
        PyClassInitializer::from(Component { spec }).add_subclass(Self::default())
    }

    /// Returns the spec that `handle` points at.
    fn spec<'a>(handle: &'a Bound<'_, Self>) -> &'a ComponentSpec {
        &handle.as_super().get().spec
    }

    /// Builds a handle from the bytes [`Component::__getstate__`] returns.
    /// Unpickling calls it through the kind's `_unpickle`.
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

/// The shared methods of a concrete class, such as `Prepend` or `Sequence`.
///
/// Each class keeps a typed pointer to its spec.
/// Its [`Component`] base holds the same spec as a [`ComponentSpec`].
pub(crate) trait ComponentType:
    PyClass<BaseType: ComponentKind> + From<Arc<<Self as ComponentType>::Spec>>
where
    ComponentSpec: From<Arc<<Self as ComponentType>::Spec>>,
{
    /// [`SequenceSpec`](component_spec::SequenceSpec) for a `Sequence` class,
    /// [`SingleSpec`](component_spec::SingleSpec) for any other.
    type Spec;

    /// Returns the pyo3 initializer over `spec`: the [`Component`] base, the kind, then this class.
    fn initializer(spec: Arc<Self::Spec>) -> PyClassInitializer<Self> {
        let base = Self::BaseType::default_initializer(ComponentSpec::from(spec.clone()));
        base.add_subclass(Self::from(spec))
    }

    /// Builds a handle on `spec`, typed as its kind.
    /// So a getter can return any class of the kind.
    fn from_spec(py: Python<'_>, spec: Arc<Self::Spec>) -> PyResult<Py<Self::BaseType>> {
        Ok(Bound::new(py, Self::initializer(spec))?
            .into_super()
            .unbind())
    }
}
