//! [`ComponentSequence`], the logic of each kind's `Sequence` class, such as
//! `normalizers.Sequence`.
//!
//! pyo3 can't make a generic class, so each kind needs its own `Sequence` class.
//! [`sequence_class!`](macro@super::macros::sequence_class) writes these classes.
//! pyo3 methods defer logic to this trait. Logic is written once for each kind.
//!
//! The trait does three things:
//!
//! - It builds a sequence from a list of items. Nested Sequence get flattened.
//! - Indexes the sequence, with support for python negative indices.
//! - Getters and setters for an index.

use std::sync::Arc;

use pyo3::PyClass;
use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;

use crate::components::ComponentKind;
use crate::components::component_spec::{ComponentSpec, SequenceSpec};
use crate::error::err;

/// The logic of the `Sequence` class of the kind `K`.
pub(crate) trait ComponentSequence<K: ComponentKind>:
    PyClass<BaseType = K> + Default
{
    /// Returns the pyo3 initializer of a `Sequence` of `items`. An item that is itself a
    /// `Sequence` adds its own items, so the sequence stays flat.
    fn initializer(items: &[Bound<'_, K>]) -> PyClassInitializer<Self> {
        let mut flat = Vec::with_capacity(items.len());
        for item in items {
            match K::spec(item) {
                ComponentSpec::Single(single) => flat.push(single.clone()),
                ComponentSpec::Sequence(sequence) => flat.extend(sequence.items()),
            }
        }
        let items_key = K::ITEMS_KEY.expect("only a kind with an items key has a Sequence");
        let spec = ComponentSpec::Sequence(Arc::new(SequenceSpec::new(items_key, flat)));
        K::initializer(spec, Self::default())
    }

    /// Returns the spec of `sequence`.
    fn spec<'a>(sequence: &'a Bound<'_, Self>) -> &'a Arc<SequenceSpec> {
        let ComponentSpec::Sequence(spec) = K::spec(sequence.as_super()) else {
            unreachable!("a Sequence is built only on a SequenceSpec, by from_spec or new");
        };
        spec
    }

    /// Returns a handle on the item at `index`. A negative `index` counts from the end.
    fn item(sequence: &Bound<'_, Self>, index: isize) -> PyResult<Py<K>> {
        let spec = Self::spec(sequence);
        let Some(item) = spec.index(index).and_then(|index| spec.get(index)) else {
            return Err(PyIndexError::new_err("Sequence index out of range"));
        };
        K::from_spec(sequence.py(), item)
    }

    /// Replaces the item at `index` with `item`'s spec. A negative `index` counts from the end.
    fn set_item(sequence: &Bound<'_, Self>, index: isize, item: &Bound<'_, K>) -> PyResult<()> {
        let Some(item) = K::spec(item).as_single() else {
            return Err(err("a Sequence item can't be a Sequence"));
        };
        let spec = Self::spec(sequence);
        spec.index(index)
            .and_then(|index| spec.set(index, item.clone()))
            .ok_or_else(|| PyIndexError::new_err("Sequence assignment index out of range"))
    }
}
