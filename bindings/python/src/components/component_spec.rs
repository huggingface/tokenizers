//! A component's spec: the component's JSON object in `tokenizer.json`.
//!
//! A Python component object is a *handle* holding an `Arc<ComponentSpec>`. It holds no data of
//! its own, so an edit through one handle reaches every handle on the same spec:
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
//! A `Sequence` runs several components of one kind in turn. It keeps one `Arc<ComponentSpec>` per
//! item, so an item and the object the sequence was built from are *the same spec*: editing either
//! changes both.
//!
//! ```text
//!   tok.normalizer ──▶ ComponentSpec {"type": "Sequence"} ─┬─▶ ComponentSpec {"type": "Prepend", ...}
//!                                                          └─▶ ComponentSpec {"type": "NFC"}
//!                            ▲
//!   tok.normalizer[1] ───────┘  the same Arc, not a copy
//! ```
//!
//! Its JSON is therefore rebuilt from its items on each read.
//!
//! # One type, one rule
//!
//! A component is a `Sequence` of its items. A single component is the degenerate case: a sequence
//! of one, whose item is the JSON it holds (`items_key: None`). Both are [`ComponentSpec`]:
//!
//! ```text
//!                          items_key        items     json()
//! ─────────────────────────────────────────────────────────────────────────────
//! Prepend("a")             None             []        {"type": "Prepend", ...}
//! Sequence([Prepend("a")]) "normalizers"    [p]       {"type": "Sequence", "normalizers": [p]}
//! Sequence([])             "normalizers"    []        {"type": "Sequence", "normalizers": []}
//! ```
//!
//! `items_key.is_some()` is the whole discriminator. Nothing else in the spec distinguishes the two.
//!
//! Its JSON is therefore rebuilt from its items on each read.
//!
//! # Validation
//!
//! A spec holds JSON as it is: `Prepend("x")` writes `{"type": "Prepend", "prepend": "x"}` and
//! nothing checks it. [`compile`](crate::tokenizer::Tokenizer::compile) is the only place that
//! builds anything, through `tk::from_json`, which validates and refuses what it can not build.

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use pyo3::prelude::*;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::JsonObject;
use crate::error::err;

/// One component's JSON, and the items when it is a `Sequence`.
///
/// Handles share it through an `Arc`. Use [`deep_clone`](Self::deep_clone) for an independent copy.
pub(crate) struct ComponentSpec {
    /// The component's JSON. For a `Sequence`, everything but its items.
    json: Mutex<Arc<JsonObject>>,
    /// The key a `Sequence`'s items live under in `tokenizer.json`, such as `"normalizers"`.
    /// `None` for any other component.
    items_key: Option<&'static str>,
    /// The items of a `Sequence`, shared with the objects it was built from.
    items: Mutex<Vec<Arc<ComponentSpec>>>,
}

impl ComponentSpec {
    pub(crate) fn new(json: JsonObject) -> Arc<Self> {
        Arc::new(Self {
            json: Mutex::new(Arc::new(json)),
            items_key: None,
            items: Mutex::new(Vec::new()),
        })
    }

    /// Builds a `Sequence` over `items`, which it shares.
    pub(crate) fn sequence(items_key: &'static str, items: Vec<Arc<ComponentSpec>>) -> Arc<Self> {
        let mut json = JsonObject::new();
        json.insert("type".to_owned(), "Sequence".into());
        Arc::new(Self {
            json: Mutex::new(Arc::new(json)),
            items_key: Some(items_key),
            items: Mutex::new(items),
        })
    }

    /// Builds a spec from a component's JSON object.
    ///
    /// A `Sequence` splits into one spec per item, so that a handle on an item can be built and
    /// edited. A JSON that does not split is kept as it is, for `compile()` to refuse.
    pub(crate) fn from_json(items_key: Option<&'static str>, mut json: JsonObject) -> Arc<Self> {
        let is_sequence = json.get("type").and_then(Value::as_str) == Some("Sequence");
        let Some(key) = items_key.filter(|_| is_sequence) else {
            return Self::new(json);
        };
        let items = match json.get(key) {
            Some(Value::Array(items)) => items
                .iter()
                .map(|item| {
                    item.as_object()
                        .map(|item| Self::from_json(Some(key), item.clone()))
                })
                .collect::<Option<Vec<_>>>(),
            _ => None,
        };
        let Some(items) = items else {
            return Self::new(json);
        };
        json.remove(key);
        Arc::new(Self {
            json: Mutex::new(Arc::new(json)),
            items_key: Some(key),
            items: Mutex::new(items),
        })
    }

    /// Returns the current JSON. A later edit leaves the returned value unchanged.
    pub(crate) fn json(&self) -> Arc<JsonObject> {
        let Some(key) = self.items_key else {
            return self.base();
        };
        let mut json = Arc::unwrap_or_clone(self.base());
        let items = self
            .items()
            .iter()
            .map(|item| Value::Object(Arc::unwrap_or_clone(item.json())))
            .collect();
        json.insert(key.to_owned(), Value::Array(items));
        Arc::new(json)
    }

    /// Returns an independent copy. It shares the JSON until either one is edited.
    pub(crate) fn deep_clone(&self) -> Arc<Self> {
        Arc::new(Self {
            json: Mutex::new(self.base()),
            items_key: self.items_key,
            items: Mutex::new(self.items().iter().map(|item| item.deep_clone()).collect()),
        })
    }

    /// Runs `edit` on the JSON. It copies the JSON first if another spec shares it.
    fn edit<R>(&self, edit: impl FnOnce(&mut JsonObject) -> R) -> R {
        edit(Arc::make_mut(&mut self.lock()))
    }

    /// Reads the `key` field as a `T`. A missing key reads as `null`, so an `Option<T>` gets
    /// `None`.
    pub(crate) fn field<T: DeserializeOwned>(&self, key: &str) -> PyResult<T> {
        let json = self.json();
        let value = json.get(key).unwrap_or(&Value::Null);
        T::deserialize(value).map_err(|e| err(format!("`{key}`: {e}")))
    }

    /// Sets the `key` field. Every handle on this spec sees the new value.
    pub(crate) fn set_field(&self, key: &str, value: impl Into<Value>) {
        self.edit(|json| json.insert(key.to_owned(), value.into()));
    }

    /// Returns the component's `"type"`, such as `"Prepend"`.
    pub(crate) fn json_type(&self) -> Option<String> {
        self.base().get("type")?.as_str().map(str::to_owned)
    }

    /// The key this spec's items live under, or `None` for a component that holds its JSON as is.
    pub(crate) fn items_key(&self) -> Option<&'static str> {
        self.items_key
    }

    pub(crate) fn len(&self) -> usize {
        self.items().len()
    }

    /// Returns the item at `index`: the same spec the item was built from.
    pub(crate) fn get(&self, index: usize) -> Option<Arc<ComponentSpec>> {
        self.items().get(index).cloned()
    }

    /// Replaces the item at `index`. It returns `None` when `index` is out of range.
    pub(crate) fn set(&self, index: usize, item: Arc<ComponentSpec>) -> Option<()> {
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

    /// The JSON without the items a `Sequence` holds separately.
    fn base(&self) -> Arc<JsonObject> {
        self.lock().clone()
    }

    fn lock(&self) -> MutexGuard<'_, Arc<JsonObject>> {
        self.json.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn items(&self) -> MutexGuard<'_, Vec<Arc<ComponentSpec>>> {
        self.items.lock().unwrap_or_else(PoisonError::into_inner)
    }
}
