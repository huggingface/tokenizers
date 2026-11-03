//! A component's spec: the component's JSON representation in `tokenizer.json`.
//!
//! A Python component object is a *handle* holding a [`ComponentSpec`]. It holds no data of its
//! own, so an edit through one handle reaches every handle on the same spec:
//!
//! ```python
//! b.normalizer = a.normalizer
//! a.normalizer.prepend = "_"            # b.normalizer changes too
//! tokenizer.model.ignore_merges = True  # edits the tokenizer's model in place
//! ```
//!
//! [`ComponentSpec::deep_clone`] returns an independent copy. The two share their JSON until one is
//! edited: [`Arc::make_mut`] copies on write.
//!
//! # Sequences
//!
//! A `Sequence` runs several components of one kind in turn. Its [`SequenceSpec`] keeps one
//! `Arc<SingleSpec>` per item, so an item and the object the sequence was built from are *the same
//! spec*: editing either changes both.
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
//! An item is a [`SingleSpec`], never a sequence. A sequence holding another could end up holding
//! itself, and its JSON would never end. A file may still nest them: tk runs a nested `Sequence`
//! like its items in place, so loading flattens it.
//!
//! # Locks
//!
//! Each spec has its own `Mutex`, held for a single read or edit. A [`SequenceSpec`] keeps its lock
//! while it reads its items' JSON. A [`SingleSpec`] never locks a [`SequenceSpec`], so every thread
//! takes the two locks in the same order, and they can't deadlock.
//!
//! # Validation
//!
//! A spec holds JSON as it is: `Prepend("x")` writes `{"type": "Prepend", "prepend": "x"}` and
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
    pub(crate) fn new(json: JsonObject) -> Self {
        Self::Single(Arc::new(SingleSpec::new(json)))
    }

    /// Builds a spec from a component's JSON representation.
    ///
    /// A `Sequence` splits into one [`SingleSpec`] per item, so that a handle on an item can be
    /// built and edited. A JSON that does not split is kept as it is, for `compile()` to refuse.
    pub(crate) fn from_json(items_key: Option<&'static str>, json: JsonObject) -> Self {
        let Some(items_key) = items_key.filter(|_| is_sequence(&json)) else {
            return Self::new(json);
        };
        let mut items = Vec::new();
        match flatten_items(items_key, &json, &mut items) {
            Some(()) => Self::Sequence(Arc::new(SequenceSpec::new(items_key, items))),
            None => Self::new(json),
        }
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

    /// Returns the spec as a `Sequence` item, or an error for a `Sequence`.
    pub(crate) fn as_item(&self) -> PyResult<&Arc<SingleSpec>> {
        match self {
            Self::Single(single) => Ok(single),
            Self::Sequence(_) => Err(err("a Sequence item can't be a Sequence")),
        }
    }
}

/// Appends the items of the `Sequence` `json` to `items`, with the items of a nested `Sequence` in
/// its place. It returns `None` when an item list is not a list of objects.
fn flatten_items(
    items_key: &str,
    json: &JsonObject,
    items: &mut Vec<Arc<SingleSpec>>,
) -> Option<()> {
    let Value::Array(values) = json.get(items_key)? else {
        return None;
    };
    for value in values {
        let item = value.as_object()?;
        if is_sequence(item) {
            flatten_items(items_key, item, items)?;
        } else {
            items.push(Arc::new(SingleSpec::new(item.clone())));
        }
    }
    Some(())
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
        self.lock().clone()
    }

    /// Reads the `key` field as a `T`. A missing key reads as `null`, so an `Option<T>` gets
    /// `None`.
    pub(crate) fn field<T: DeserializeOwned>(&self, key: &str) -> PyResult<T> {
        let json = self.json();
        let value = json.get(key).unwrap_or(&Value::Null);
        T::deserialize(value).map_err(|e| err(format!("`{key}`: {e}")))
    }

    /// Sets the `key` field. Every handle on this spec sees the new value.
    /// It copies the JSON first if another [`SingleSpec`] shares it.
    pub(crate) fn set_field(&self, key: &str, value: impl Into<Value>) {
        Arc::make_mut(&mut self.lock()).insert(key.to_owned(), value.into());
    }

    /// Returns the component's `"type"`, such as `"Prepend"`.
    pub(crate) fn json_type(&self) -> Option<String> {
        self.json().get("type")?.as_str().map(str::to_owned)
    }

    fn lock(&self) -> MutexGuard<'_, Arc<JsonObject>> {
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
        self.items().len()
    }

    /// Returns the item at `index`: the same spec the item was built from.
    pub(crate) fn get(&self, index: usize) -> Option<ComponentSpec> {
        self.items().get(index).cloned().map(ComponentSpec::Single)
    }

    /// Replaces the item at `index`. It returns `None` when `index` is out of range.
    pub(crate) fn set(&self, index: usize, item: Arc<SingleSpec>) -> Option<()> {
        *self.items().get_mut(index)? = item;
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
            .items()
            .iter()
            .map(|item| Arc::new(item.deep_clone()))
            .collect();
        Self::new(self.items_key, items)
    }

    /// Builds the sequence's JSON: `"type": "Sequence"`, and the items' JSON under `items_key`.
    fn json(&self) -> JsonObject {
        let items: Vec<Value> = self
            .items()
            .iter()
            .map(|item| Value::Object(Arc::unwrap_or_clone(item.json())))
            .collect();
        let mut json = JsonObject::new();
        json.insert("type".to_owned(), "Sequence".into());
        json.insert(self.items_key.to_owned(), items.into());
        json
    }

    fn items(&self) -> MutexGuard<'_, Vec<Arc<SingleSpec>>> {
        self.items.lock().unwrap_or_else(PoisonError::into_inner)
    }
}
