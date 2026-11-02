//! The specs behind the components of a [`Tokenizer`](crate::tokenizer::Tokenizer).
//!
//! A *spec* is a component's JSON representation in `tokenizer.json`, such as:
//!
//! ```json
//! {"type": "Prepend", "prepend": "▁"}
//! ```
//!
//! The [`components`](super#handles) docs introduce tokenizer fields and handles, and show how
//! Python edits a spec.
//!
//! # Why components store JSON
//!
//! A tokenizer field holds the component's JSON, not a Rust struct.
//!
//! - **No conversion.** [`to_str`](crate::tokenizer::Tokenizer::to_str), pickle and
//!   [`compile`](crate::tokenizer::Tokenizer::compile) all need the JSON. They read it as is.
//! - **A Python class is a thin layer.** It adds one getter and setter per JSON key, such as
//!   `Prepend.prepend` for `"prepend"`. No Rust struct duplicates the core's types, so the two
//!   cannot drift apart.
//!
//! # Sharing and copying
//!
//! A tokenizer field or handle holds an `Arc<SingleSpec>`, and a [`SingleSpec`] holds its JSON as a `Mutex<Arc<JsonObject>>`.
//!
//! - **Handles share the `Arc<SingleSpec>`.**
//!   An edit through `tok.normalizer` shows in every other handle on the same spec.
//! - **Copies share the `Arc<JsonObject>`.**
//!   `copy.deepcopy(tok)` and `compile()` give the copy a new `SingleSpec` over the same JSON.
//!   Copying a tokenizer therefore copies no JSON.
//! - **An edit copies the JSON only when it is shared.**
//!   [`Arc::make_mut`] is copy-on-write.
//!   It edits the `JsonObject` in place when this `SingleSpec` is its only owner.
//!   Otherwise it copies the JSON first, so a copy keeps the JSON it was made from.
//!
//! # Sequences
//!
//! A `Sequence` runs several components of one kind in turn, such as two normalizers. Its
//! [`SequenceSpec`] holds one `Arc<SingleSpec>` per item.
//! Each item is a component like any other, with its own identity:
//!
//! - `seq[1]` returns a handle on item 1's `SingleSpec`.
//! - After `seq = Sequence([p])`, `p` and `seq[0]` share one `SingleSpec`. An edit through `p`
//!   changes the sequence too.
//!
//! ```text
//!   tok.normalizer ──▶ SequenceSpec ─┬─▶ SingleSpec ──▶ {"type": "Prepend", "prepend": "▁"}
//!                                    └─▶ SingleSpec ──▶ {"type": "NFC"}
//!                                          ▲
//!   tok.normalizer[1] ─────────────────────┘
//! ```
//!
//! [`ComponentSpec::json`] rebuilds the sequence's JSON from its items on every read.
//!
//!
//! # Locks
//!
//! Each spec has its own `Mutex`, held for a single read or edit. A `SequenceSpec` keeps its
//! lock while it reads its items' JSON. A `SingleSpec` never locks a `SequenceSpec`. So every
//! thread takes the two locks in the same order, which rules out a deadlock between them.

use std::sync::{Arc, Mutex, PoisonError};

use pyo3::prelude::*;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::JsonObject;
use crate::error::{NestedSequenceError, err};

/// A component kind, named after its top-level key in `tokenizer.json`.
///
/// Each [`ComponentKind`](super::ComponentKind) class names its key in `JSON_KEY`.
#[derive(Clone, Copy)]
pub(crate) enum ComponentKey {
    Normalizer,
    PreTokenizer,
    PostProcessor,
    Decoder,
    Model,
}

impl ComponentKey {
    /// The top-level `tokenizer.json` key of this kind, such as `"pre_tokenizer"`.
    pub(crate) fn json_key(self) -> &'static str {
        match self {
            ComponentKey::Normalizer => "normalizer",
            ComponentKey::PreTokenizer => "pre_tokenizer",
            ComponentKey::PostProcessor => "post_processor",
            ComponentKey::Decoder => "decoder",
            ComponentKey::Model => "model",
        }
    }

    /// Returns the key that lists the items of a `Sequence` of this kind:
    /// ```json
    /// "normalizer": {
    ///     "type": "Sequence",
    ///     "normalizers": [{ "type": "NFC" }]   // <-- the items key
    /// }
    /// ```
    ///
    /// Models have no `Sequence`, so they return `None`.
    pub(crate) fn sequence_items_key(self) -> Option<&'static str> {
        match self {
            ComponentKey::Normalizer => Some("normalizers"),
            ComponentKey::PreTokenizer => Some("pretokenizers"),
            ComponentKey::PostProcessor => Some("processors"),
            ComponentKey::Decoder => Some("decoders"),
            ComponentKey::Model => None,
        }
    }
}

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
    /// Builds the spec of a `kind` component from its JSON representation in `tokenizer.json`.
    ///
    /// It splits a `Sequence` into one [`SingleSpec`] per item. It returns an error for a
    /// malformed item list or a nested `Sequence`.
    pub(crate) fn from_json(kind: ComponentKey, mut json: JsonObject) -> tk::Result<Self> {
        let Some(items_key) = kind.sequence_items_key() else {
            return Ok(Arc::new(SingleSpec::new(json)).into());
        };
        if json.get("type").and_then(Value::as_str) != Some("Sequence") {
            return Ok(Arc::new(SingleSpec::new(json)).into());
        }
        let Some(Value::Array(items)) = json.remove(items_key) else {
            return Err(malformed_items(kind, items_key).into());
        };
        let mut specs = Vec::with_capacity(items.len());
        for item in items {
            let Value::Object(item) = item else {
                return Err(malformed_items(kind, items_key).into());
            };
            specs.push(Self::from_json(kind, item)?);
        }
        let sequence = SequenceSpec::new(kind, specs)
            .map_err(|error| format!("malformed {} Sequence: {error}", kind.json_key()))?;
        Ok(Arc::new(sequence).into())
    }

    /// Returns an independent copy.
    /// The copy shares the JSON until the first edit, so it costs no JSON copy.
    pub(crate) fn deep_clone(&self) -> Self {
        match self {
            Self::Single(single) => Self::Single(Arc::new(single.deep_clone())),
            Self::Sequence(sequence) => Self::Sequence(Arc::new(sequence.deep_clone())),
        }
    }

    /// Returns the component's JSON representation in `tokenizer.json`.
    /// A `Sequence` builds it from its items on each call.
    pub(crate) fn json(&self) -> Arc<JsonObject> {
        match self {
            Self::Single(single) => single.json(),
            Self::Sequence(sequence) => Arc::new(sequence.json()),
        }
    }
}

impl From<Arc<SingleSpec>> for ComponentSpec {
    fn from(single: Arc<SingleSpec>) -> Self {
        Self::Single(single)
    }
}

impl From<Arc<SequenceSpec>> for ComponentSpec {
    fn from(sequence: Arc<SequenceSpec>) -> Self {
        Self::Sequence(sequence)
    }
}

/// Unwraps a `Sequence` item. A `Sequence` cannot be an item, so it returns `NestedSequenceError`.
impl TryFrom<ComponentSpec> for Arc<SingleSpec> {
    type Error = NestedSequenceError;

    fn try_from(spec: ComponentSpec) -> Result<Self, NestedSequenceError> {
        match spec {
            ComponentSpec::Single(single) => Ok(single),
            ComponentSpec::Sequence(_) => Err(NestedSequenceError),
        }
    }
}

/// The spec of one component that is not a `Sequence`.
///
/// Its JSON is copied on write: see [Sharing and copying](self#sharing-and-copying).
pub(crate) struct SingleSpec {
    json: Mutex<Arc<JsonObject>>,
}

impl SingleSpec {
    pub(crate) fn new(json: JsonObject) -> Self {
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
    pub(crate) fn json(&self) -> Arc<JsonObject> {
        self.json
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Runs `edit` on the JSON. It copies the JSON first if another [`SingleSpec`] shares it.
    fn edit<R>(&self, edit: impl FnOnce(&mut JsonObject) -> R) -> R {
        edit(Arc::make_mut(
            &mut self.json.lock().unwrap_or_else(PoisonError::into_inner),
        ))
    }

    /// Reads the `key` field as a `T`. A missing key reads as `null`, so an `Option<T>` gets
    /// `None`.
    pub(crate) fn field<T: DeserializeOwned>(&self, key: &str) -> PyResult<T> {
        let json = self.json();
        let value = json.get(key).unwrap_or(&Value::Null);
        T::deserialize(value).map_err(|e| err(format!("`{key}`: {e}")))
    }

    /// Sets the `key` field. Every tokenizer field and handle on this spec sees the new value.
    pub(crate) fn set_field(&self, key: &str, value: impl Into<Value>) {
        self.edit(|json| json.insert(key.to_owned(), value.into()));
    }

    /// Returns the component's `"type"`, such as `"Prepend"`.
    pub(crate) fn json_type(&self) -> Option<String> {
        self.json().get("type")?.as_str().map(str::to_owned)
    }
}

/// The spec of a `Sequence`: one shared [`SingleSpec`] per item.
///
/// A handle on an item writes through to the sequence: see [Sequences](self#sequences).
pub(crate) struct SequenceSpec {
    /// See [`ComponentKey::sequence_items_key`].
    items_key: &'static str,
    items: Mutex<Vec<Arc<SingleSpec>>>,
}

impl SequenceSpec {
    /// Builds a `Sequence` of `kind` from its items.
    /// It returns [`NestedSequence`] if an item is a `Sequence`.
    pub(crate) fn new(
        kind: ComponentKey,
        items: Vec<ComponentSpec>,
    ) -> Result<Self, NestedSequenceError> {
        let items_key = kind
            .sequence_items_key()
            .expect("only the kinds that have a Sequence class build a SequenceSpec");
        let mut singles = Vec::with_capacity(items.len());
        for item in items {
            singles.push(Arc::try_from(item)?);
        }
        Ok(Self {
            items_key,
            items: Mutex::new(singles),
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.items
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    }

    /// Returns a spec that shares the item at `index`, or `None` when `index` is out of range.
    pub(crate) fn get(&self, index: usize) -> Option<ComponentSpec> {
        let items = self.items.lock().unwrap_or_else(PoisonError::into_inner);
        items.get(index).cloned().map(ComponentSpec::Single)
    }

    /// Replaces the item at `index`, or returns `None` when `index` is out of range.
    pub(crate) fn set(&self, index: usize, item: Arc<SingleSpec>) -> Option<()> {
        let mut items = self.items.lock().unwrap_or_else(PoisonError::into_inner);
        *items.get_mut(index)? = item;
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

    /// Returns a new [`SequenceSpec`] that holds a [`SingleSpec::deep_clone`] of each item.
    fn deep_clone(&self) -> Self {
        let items = self.items.lock().unwrap_or_else(PoisonError::into_inner);
        Self {
            items_key: self.items_key,
            items: Mutex::new(
                items
                    .iter()
                    .map(|item| Arc::new(item.deep_clone()))
                    .collect(),
            ),
        }
    }

    /// Builds the sequence's JSON representation: `"type": "Sequence"`, and the items' JSON under
    /// `items_key`.
    fn json(&self) -> JsonObject {
        let items: Vec<Value> = self
            .items
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .map(|item| Value::Object(Arc::unwrap_or_clone(item.json())))
            .collect();
        let mut json = JsonObject::new();
        json.insert("type".to_owned(), "Sequence".into());
        json.insert(self.items_key.to_owned(), items.into());
        json
    }
}

fn malformed_items(kind: ComponentKey, items_key: &str) -> String {
    let kind = kind.json_key();
    format!("malformed {kind} Sequence: the \"{items_key}\" field must be a list of {kind} objects")
}
