//! How a component's spec is stored, shared and copied.
//!
//! The [`components`](super) docs say what a spec and a handle are.
//!
//! # Why a spec is JSON
//!
//! A spec holds the component's JSON, not a Rust struct:
//!
//! - **No conversion.** [`to_str`](crate::tokenizer::Tokenizer::to_str), pickle and
//!   [`compile`](crate::tokenizer::Tokenizer::compile) all need the JSON. They read it as is.
//! - **Thin Python classes.** A class adds one getter and setter per JSON key, such as
//!   `Prepend.prepend` for `"prepend"`. No Rust struct copies the core's types, so the two can't
//!   drift apart.
//!
//! # The pointers
//!
//! A tokenizer field and a handle hold clones of the same [`ComponentSpec`]. Cloning it copies a
//! pointer, not the JSON:
//!
//! ```text
//!   Tokenizer ──▶ TokenizerSpec ──▶ normalizer: ComponentSpec ──┐
//!                                                               ├──▶ SingleSpec ──▶ JSON
//!   tok.normalizer ──▶ Component ──▶ spec: ComponentSpec ───────┘
//! ```
//!
//! A [`ComponentSpec`] is one of two variants:
//!
//! - `Single(Arc<SingleSpec>)` is one component. [`SingleSpec`] holds its JSON.
//! - `Sequence(Arc<SequenceSpec>)` is a `Sequence`. [`SequenceSpec`] holds one `Arc<SingleSpec>`
//!   per item.
//!
//! # Two kinds of `Arc`
//!
//! Each [`SingleSpec`] sits behind two `Arc`s, and each shares something different:
//!
//! - **`Arc<SingleSpec>` shares the spec.** Every handle on it holds this pointer. An edit
//!   through one handle shows in all of them.
//! - **`Arc<JsonObject>` shares the JSON.** `copy.deepcopy` and `compile()` build a new
//!   `SingleSpec` over the same JSON. So copying a tokenizer copies no JSON, however big its vocab.
//!
//! An edit goes through [`Arc::make_mut`]. When this `SingleSpec` is the JSON's only owner, it
//! edits the JSON in place. Otherwise it copies the JSON first, so the copy keeps the JSON it was
//! made from. This is called *copy on write*.
//!
//! # Sequences
//!
//! A `Sequence` runs several components of one kind in turn. Its [`SequenceSpec`] holds the same
//! `Arc<SingleSpec>` as the object each item came from, so editing either one changes both:
//!
//! ```text
//!   tok.normalizer ──▶ SequenceSpec ─┬─▶ SingleSpec {"type": "Prepend", "prepend": "▁"}
//!                                    └─▶ SingleSpec {"type": "NFC"}
//!                                          ▲
//!   tok.normalizer[1] ─────────────────────┘  the same Arc, not a copy
//! ```
//!
//! [`ComponentSpec::json`] rebuilds the sequence's JSON from its items on every read.
//!
//! # No nested sequences
//!
//! An item is always a [`SingleSpec`], never a sequence. A sequence that held another could end
//! up holding itself, and its JSON would never end.
//!
//! - Loading a file flattens a nested `Sequence` into its items.
//! - `Sequence([...])` does the same with a `Sequence` argument.
//! - `seq[i] = x` refuses a `Sequence`, since splicing it in would change the length.
//!
//! tk runs a nested normalizer or decoder `Sequence` like its items, so flattening keeps the ids.
//!
//! # Locks
//!
//! Three kinds of `Mutex` guard a tokenizer, taken in this order:
//!
//! 1. The [`TokenizerSpec`](crate::tokenizer::TokenizerSpec), for which spec each field holds.
//! 2. A [`SequenceSpec`], for its items.
//! 3. A [`SingleSpec`], for its JSON.
//!
//! Two rules keep them from deadlocking:
//!
//! - **One order.** Code takes a lock only after the ones above it, never before. A
//!   `SequenceSpec` holds its lock while it reads its items' JSON, and a `SingleSpec` never locks
//!   anything else.
//! - **No Python under a lock.** Code drops its guard before it calls into Python. Building a
//!   Python object can run any Python code, and that code may need the same lock.
//!
//! # Validation
//!
//! A spec holds JSON as it is: `Prepend("x")` writes `{"type": "Prepend", "prepend": "x"}`, and
//! nothing checks it. [`compile`](crate::tokenizer::Tokenizer::compile) is the only place that
//! builds anything, through `tk::from_json`, which refuses what it can't build.

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use pyo3::prelude::*;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::JsonObject;
use crate::error::err;

/// The spec of one component, or of a `Sequence`.
///
/// A clone shares the spec, so an edit through either one reaches both.
/// [`deep_clone`](Self::deep_clone) returns an independent copy.
#[derive(Clone)]
pub(crate) enum ComponentSpec {
    Single(Arc<SingleSpec>),
    Sequence(Arc<SequenceSpec>),
}

impl ComponentSpec {
    /// Builds a spec that holds `json` as is, without splitting it into items.
    pub(crate) fn single(json: JsonObject) -> Self {
        Self::Single(Arc::new(SingleSpec::new(json)))
    }

    /// Builds a spec from a component's JSON representation.
    ///
    /// A `Sequence` splits into one [`SingleSpec`] per item, so that a handle on an item can be
    /// built and edited. A `Sequence` whose items are not a list of objects is kept as it is, for
    /// `compile()` to refuse.
    pub(crate) fn from_json(items_key: Option<&'static str>, json: JsonObject) -> Self {
        if let Some(items_key) = items_key
            && is_sequence(&json)
            && let Some(items) = sequence_items(items_key, &json)
        {
            return Self::Sequence(Arc::new(SequenceSpec::new(items_key, items)));
        }
        Self::single(json)
    }

    /// Returns the component's JSON representation in `tokenizer.json`.
    /// A later edit leaves the returned value unchanged.
    pub(crate) fn json(&self) -> Arc<JsonObject> {
        match self {
            Self::Single(single) => single.json(),
            Self::Sequence(sequence) => Arc::new(sequence.json()),
        }
    }

    /// Returns an independent copy. It shares the JSON until either one is edited.
    pub(crate) fn deep_clone(&self) -> Self {
        match self {
            Self::Single(single) => Self::Single(Arc::new(single.deep_clone())),
            Self::Sequence(sequence) => Self::Sequence(Arc::new(sequence.deep_clone())),
        }
    }

    pub(crate) fn as_single(&self) -> Option<&Arc<SingleSpec>> {
        match self {
            Self::Single(single) => Some(single),
            Self::Sequence(_) => None,
        }
    }
}

/// Returns the items of the `Sequence` `json`, with a nested `Sequence` replaced by its items.
/// It returns `None` when an item list is not a list of objects.
fn sequence_items(items_key: &str, json: &JsonObject) -> Option<Vec<Arc<SingleSpec>>> {
    let Value::Array(values) = json.get(items_key)? else {
        return None;
    };
    let mut items = Vec::with_capacity(values.len());
    for value in values {
        let item = value.as_object()?;
        if is_sequence(item) {
            items.extend(sequence_items(items_key, item)?);
        } else {
            items.push(Arc::new(SingleSpec::new(item.clone())));
        }
    }
    Some(items)
}

fn is_sequence(json: &JsonObject) -> bool {
    json.get("type").and_then(Value::as_str) == Some("Sequence")
}

/// The spec of one component that is not a `Sequence`.
pub(crate) struct SingleSpec {
    json: Mutex<Arc<JsonObject>>,
}

impl SingleSpec {
    fn new(json: JsonObject) -> Self {
        Self {
            json: Mutex::new(Arc::new(json)),
        }
    }

    /// Returns a new [`SingleSpec`] that shares this JSON until either one is edited.
    fn deep_clone(&self) -> Self {
        Self {
            json: Mutex::new(self.json()),
        }
    }

    /// Returns the current JSON. A later edit leaves the returned value unchanged.
    fn json(&self) -> Arc<JsonObject> {
        self.lock_json().clone()
    }

    /// Reads the `key` field, as [`JsonFieldsExt::field`](super::JsonFieldsExt::field) describes.
    pub(crate) fn field<T: DeserializeOwned>(&self, key: &str) -> PyResult<T> {
        let json = self.json();
        let value = json.get(key).unwrap_or(&Value::Null);
        T::deserialize(value).map_err(|e| err(format!("`{key}`: {e}")))
    }

    /// Sets the `key` field, as [`JsonFieldsExt::set_field`](super::JsonFieldsExt::set_field)
    /// describes. It copies the JSON first if another [`SingleSpec`] shares it.
    pub(crate) fn set_field(&self, key: &str, value: impl Into<Value>) {
        Arc::make_mut(&mut self.lock_json()).insert(key.to_owned(), value.into());
    }

    /// Returns the component's `"type"`, such as `"Prepend"`.
    pub(crate) fn json_type(&self) -> Option<String> {
        self.json().get("type")?.as_str().map(str::to_owned)
    }

    fn lock_json(&self) -> MutexGuard<'_, Arc<JsonObject>> {
        self.json.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// The spec of a `Sequence`: one shared [`SingleSpec`] per item.
pub(crate) struct SequenceSpec {
    /// The key the items live under in `tokenizer.json`, such as `"normalizers"`.
    items_key: &'static str,
    items: Mutex<Vec<Arc<SingleSpec>>>,
}

impl SequenceSpec {
    pub(crate) fn new(items_key: &'static str, items: Vec<Arc<SingleSpec>>) -> Self {
        Self {
            items_key,
            items: Mutex::new(items),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.lock_items().len()
    }

    /// Returns the item at `index`: the same spec the item was built from.
    pub(crate) fn get(&self, index: usize) -> Option<ComponentSpec> {
        self.lock_items()
            .get(index)
            .cloned()
            .map(ComponentSpec::Single)
    }

    /// Replaces the item at `index`. It returns `None` when `index` is out of range.
    pub(crate) fn set(&self, index: usize, item: Arc<SingleSpec>) -> Option<()> {
        *self.lock_items().get_mut(index)? = item;
        Some(())
    }

    /// Converts a Python index into a position. A negative index counts from the end.
    /// It returns `None` when `index` is out of range.
    pub(crate) fn index(&self, index: isize) -> Option<usize> {
        let len = self.len();
        let index = match usize::try_from(index) {
            Ok(index) => index,
            Err(_) => len.checked_sub(index.unsigned_abs())?,
        };
        (index < len).then_some(index)
    }

    /// Returns a new [`SequenceSpec`] holding a [`SingleSpec::deep_clone`] of each item.
    fn deep_clone(&self) -> Self {
        let items: Vec<Arc<SingleSpec>> = self
            .lock_items()
            .iter()
            .map(|item| Arc::new(item.deep_clone()))
            .collect();
        Self::new(self.items_key, items)
    }

    /// Builds the sequence's JSON: `"type": "Sequence"`, and the items' JSON under `items_key`.
    fn json(&self) -> JsonObject {
        let items: Vec<Value> = self
            .lock_items()
            .iter()
            .map(|item| Value::Object(Arc::unwrap_or_clone(item.json())))
            .collect();
        let mut json = JsonObject::new();
        json.insert("type".to_owned(), "Sequence".into());
        json.insert(self.items_key.to_owned(), items.into());
        json
    }

    pub(crate) fn lock_items(&self) -> MutexGuard<'_, Vec<Arc<SingleSpec>>> {
        self.items.lock().unwrap_or_else(PoisonError::into_inner)
    }
}
