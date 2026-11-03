//! The Python classes behind `tokenizer.model`, `tokenizer.normalizer` and the other components.
//!
//! The [`tokenizer`](crate::tokenizer) docs explain the tokenizer as a whole.
//!
//! The [`component_spec`] docs explain how a component's JSON is stored, shared and copied,
//! in a [`ComponentSpec`].
//!
//! [`Tokenizer`]: crate::tokenizer::Tokenizer
//!
//! # Specs
//!
//! A *spec* is one component's JSON, as it appears in `tokenizer.json`:
//!
//! ```json
//! {"type": "Prepend", "prepend": "▁"}
//! ```
//!
//! [`TokenizerSpec`] holds one spec per component field: `normalizer`, `pre_tokenizer`,
//! `post_processor`, `decoder` and `model`.
//!
//! [`TokenizerSpec`]: crate::tokenizer::TokenizerSpec
//!
//! # Handles
//!
//! A Python component object, such as `tokenizer.normalizer`, is a *handle*. It holds no settings
//! of its own. It points at a spec.
//!
//! - Reading `tokenizer.normalizer` returns a handle on the tokenizer's own spec. No copy.
//! - Assigning `tokenizer.normalizer = n` points the tokenizer at `n`'s spec. No copy either.
//! - `copy.deepcopy(n)` returns a handle on a new, independent spec.
//!
//! So an edit through one handle reaches the tokenizer and every other handle on the same spec:
//!
//! ```python
//! b.normalizer = a.normalizer
//! a.normalizer.prepend = "_"  # b.normalizer changes too
//! ```
//!
//! # Editing in place
//!
//! Since a getter returns a handle, not a copy, one line edits a loaded tokenizer at any depth:
//!
//! ```python
//! tokenizer.normalizer[1].prepend = "_"
//! ```
//!
//! # Python class hierarchy
//!
//! ```text
//! Component                           base class for all tokenizer components
//!  ⤷ Model | Normalizer | ...         component kinds
//!    ⤷ Prepend | Sequence | ...       concrete classes
//! ```
//!
//! - [`Component`] is the base of every component. It gives them `repr` and pickling.
//! - A *kind*, such as [`Normalizer`](normalizers::Normalizer), is the type a tokenizer field
//!   takes: `tokenizer.normalizer = x` accepts any `Normalizer`. The [`ComponentKind`] trait holds
//!   the code all kinds share.
//! - A *concrete class*, such as [`Prepend`](normalizers::Prepend), is one component type. Its
//!   constructor builds the JSON, and its properties read and write the JSON keys, such as
//!   `Prepend.prepend`.
//!
//! Each read builds a new Python object, so `tokenizer.normalizer is tokenizer.normalizer` is
//! `False`. Both objects still edit the same spec.

pub(crate) mod component_spec;
pub(crate) mod decoders;
pub(crate) mod models;
pub(crate) mod normalizers;
pub(crate) mod pre_tokenizers;
pub(crate) mod processors;
pub(crate) mod sequence_component;
pub(crate) mod stateless_component;

use std::sync::Arc;

use pyo3::PyClass;
use pyo3::impl_::pyclass::PyClassBaseType;
use pyo3::prelude::*;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::JsonObject;
use crate::components::component_spec::{ComponentSpec, SequenceSpec, SingleSpec};
use crate::error::err;
use crate::repr;
use crate::tokenizer::TokenizerSpec;

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
    /// The key a `Sequence` of this kind lists its items under, such as `"normalizers"`.
    /// `None` for a kind with no `Sequence`.
    const ITEMS_KEY: Option<&'static str>;

    /// The [`TokenizerSpec`] field holding this kind's spec.
    fn tokenizer_field_mut(spec: &mut TokenizerSpec) -> &mut Option<ComponentSpec>;

    /// Builds a handle on `spec`, of the concrete class its `"type"` names.
    fn from_spec(py: Python<'_>, spec: ComponentSpec) -> PyResult<Py<Self>>;

    /// Returns the pyo3 initializer of the plain kind over `spec`.
    /// [`Self::initializer`] adds a concrete class on top of it.
    fn default_initializer(spec: ComponentSpec) -> PyClassInitializer<Self> {
        PyClassInitializer::from(Component { spec }).add_subclass(Self::default())
    }

    /// Returns the pyo3 initializer of the concrete class `C` on `spec`.
    fn initializer<C: PyClass<BaseType = Self>>(
        spec: ComponentSpec,
        class: C,
    ) -> PyClassInitializer<C> {
        Self::default_initializer(spec).add_subclass(class)
    }

    /// Builds a handle on `spec` of the plain kind, with no concrete class.
    fn default_component(py: Python<'_>, spec: ComponentSpec) -> PyResult<Py<Self>> {
        Py::new(py, Self::default_initializer(spec))
    }

    /// Builds a handle on `spec` of the concrete class `C`, typed as its kind.
    fn concrete_component<C: PyClass<BaseType = Self>>(
        py: Python<'_>,
        spec: ComponentSpec,
        class: C,
    ) -> PyResult<Py<Self>> {
        let handle = Bound::new(py, Self::initializer(spec, class))?;
        // `into_super` changes only the Rust type, from `C` to the kind, so one getter can return
        // any of the kind's classes. The Python object stays a `C`, for `isinstance` and methods.
        Ok(handle.into_super().unbind())
    }

    /// Returns the spec that `handle` points at.
    fn spec<'a>(handle: &'a Bound<'_, Self>) -> &'a ComponentSpec {
        &handle.as_super().get().spec
    }

    /// Returns the spec of `handle`, a handle of a concrete class with JSON fields.
    fn single_spec<'a, C: PyClass<BaseType = Self>>(
        handle: &'a Bound<'_, C>,
    ) -> &'a Arc<SingleSpec> {
        let ComponentSpec::Single(single) = Self::spec(handle.as_super()) else {
            unreachable!(
                "a class with JSON fields is built only on a SingleSpec, by from_spec or new"
            );
        };
        single
    }

    /// Returns the spec of `handle`, a handle of this kind's `Sequence` class.
    fn sequence_spec<'a, C: PyClass<BaseType = Self>>(
        handle: &'a Bound<'_, C>,
    ) -> &'a Arc<SequenceSpec> {
        let ComponentSpec::Sequence(sequence) = Self::spec(handle.as_super()) else {
            unreachable!("a Sequence handle is built only on a SequenceSpec, by from_spec or new");
        };
        sequence
    }

    /// Builds a handle from the bytes [`Component::__getstate__`] returns.
    /// Unpickling calls it through the kind's `_unpickle`.
    fn from_state(py: Python<'_>, state: &[u8]) -> PyResult<Py<Self>> {
        let json: JsonObject = serde_json::from_slice(state).map_err(err)?;
        Self::from_spec(py, ComponentSpec::from_json(Self::ITEMS_KEY, json))
    }
}

/// Reads and writes the JSON fields of a concrete class's handle, such as `Prepend.prepend`.
pub(crate) trait JsonFieldsExt {
    /// Reads the `key` field as a `T`. A missing key reads as `null`, so an `Option<T>` gets
    /// `None`.
    fn field<T: DeserializeOwned>(&self, key: &str) -> PyResult<T>;
    /// Sets the `key` field. Every handle on this spec sees the new value.
    fn set_field(&self, key: &str, value: impl Into<Value>);
}

impl<K: ComponentKind, C: PyClass<BaseType = K>> JsonFieldsExt for Bound<'_, C> {
    fn field<T: DeserializeOwned>(&self, key: &str) -> PyResult<T> {
        K::single_spec(self).field(key)
    }

    fn set_field(&self, key: &str, value: impl Into<Value>) {
        K::single_spec(self).set_field(key, value);
    }
}
