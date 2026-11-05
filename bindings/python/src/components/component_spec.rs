//! Components spec storage structs.
//!
//! See [`components`](super) docs for more details about Python classes and the component hierarchy.
//!
//! # Storing components as JSON
//!
//! The Components' and Tokenizer's specs are stored as JSON objects, not rust sructs.
//! This design choice has several upsides:
//!
//! - **Cheap serialization.** [`to_str`](crate::tokenizer::Tokenizer::to_str), pickle and
//!   [`compile`](crate::tokenizer::Tokenizer::compile) all use the JSON as-is.
//! - **Thin Python classes.** A class adds one getter and setter per JSON key, such as
//!   `Prepend.prepend` for `"prepend"`. The Python class don't have to mirror the Rust core's
//!   struct, which makes it easier to maintain.
//!
//! # [`SingleSpec`] and [`SequenceSpec`]
//!
//! A [`ComponentSpec`] can be one of two variants:
//!
//! - A [`SingleSpec`] is one component, such as `Prepend`. It holds the component's JSON.
//! - A [`SequenceSpec`] is a `Sequence`, which runs several components of one kind in turn. It
//!   holds one `SingleSpec` per item.
//!
//! Splitting a `Sequence` into items lets `tok.normalizer[1]` return the item itself, so an edit
//! through it reaches the sequence. [`ComponentSpec::json`] builds the sequence's JSON from its
//! items on every read, so it always shows their latest edits.
//!
//! # Arc pointers make copies cheap
//!
//! A [`ComponentSpec`] is an `Arc` pointer over a component's JSON value. So manipulating and
//! copying Python objects is cheap: the JSON value only gets duplicated on edit.
//!
//! ```text
//!   TokenizerSpec.normalizer ──┐
//!                              ├──▶ SingleSpec ──┐
//!   tok.normalizer ────────────┘                 ├──▶ JSON
//!   copy.deepcopy(tok.normalizer) ──▶ SingleSpec ┘
//! ```
//!
//! - Reading `tok.normalizer` clones the tokenizer's own `ComponentSpec`. Both point at the same
//!   [`SingleSpec`], so an edit through one shows in the other.
//! - `copy.deepcopy` and `compile()` build a new `SingleSpec` over the same JSON. So copying a
//!   tokenizer copies no JSON, however big its vocab.
//!
//! The first edit after a copy duplicates the JSON, then changes it. The other side keeps the
//! JSON it had. This is called *copy on write*, and [`Arc::make_mut`] does it.
//!
//! # Locks
//!
//! Three kinds of `Mutex` guard a tokenizer, taken in this order:
//!
//! 1. In the [`Tokenizer`](crate::tokenizer::Tokenizer), guarding its
//!    [`TokenizerSpec`](crate::tokenizer::TokenizerSpec).
//! 2. In a [`SequenceSpec`], guarding its items.
//! 3. In a [`SingleSpec`], guarding its JSON.
//!
//! The locks are there because every Python class must be `Sync`, so that two threads can use the
//! same object safely:
//!
//! - pyo3 requires it. Any Python thread can reach any object, so pyo3 can't promise that only
//!   one thread uses it.
//! - Free-threaded Python (3.14t) has no GIL, so two threads really can edit one component at the
//!   same time.
//!
//! An editable spec is only `Sync` behind a lock, hence the `Mutex`.
//!
//! ## Rules to avoid deadlocks
//!
//! Each lock is taken by a private method of its type: `Tokenizer::spec`,
//! `SequenceSpec::lock_items` and `SingleSpec::lock_json`. Code outside these types can't hold a
//! guard. Inside them, follow three rules:
//!
//! 1. **Take locks in the order above.** While you hold one, take only locks further down the
//!    list. `SequenceSpec::json` holds its items while it locks each item's JSON, which is fine.
//! 2. **Never take a lock you already hold.** A `Mutex` is not reentrant, so a thread that locks
//!    it twice waits on itself forever. For example, don't call `self.len()` while holding
//!    `lock_items`.
//! 3. **Never call into Python while you hold a guard.** Building a Python object can run any
//!    Python code, and that code may need the same lock. Copy what you need, drop the guard, then
//!    call Python, as `Tokenizer::get` does.
//!
//! Keep the lock methods private. When code elsewhere needs what a lock guards, add a method that
//! copies it out, like [`SequenceSpec::items`].
//!
//! # Validation
//!
//! A spec holds JSON as it is: `Prepend("x")` writes `{"type": "Prepend", "prepend": "x"}`.
//! Validation is delegated to [`compile`](crate::tokenizer::Tokenizer::compile) which calls
//! `tk::from_json`.

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

    /// Builds a spec from a `json!` object literal, as the Python constructors write it.
    pub(crate) fn from_literal(json: Value) -> Self {
        let Value::Object(json) = json else {
            unreachable!("callers pass a json! object literal, got {json}")
        };
        Self::single(json)
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

    /// Returns whether the JSON has the `key` field.
    pub(crate) fn has_field(&self, key: &str) -> bool {
        self.lock_json().contains_key(key)
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

    /// Returns a copy of the item list, so the caller never holds the lock. The items are the
    /// same specs the sequence holds, not copies.
    pub(crate) fn items(&self) -> Vec<Arc<SingleSpec>> {
        self.lock_items().clone()
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

    fn lock_items(&self) -> MutexGuard<'_, Vec<Arc<SingleSpec>>> {
        self.items.lock().unwrap_or_else(PoisonError::into_inner)
    }
}
