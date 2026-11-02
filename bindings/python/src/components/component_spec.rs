use std::sync::{Arc, Mutex, PoisonError};

use pyo3::prelude::*;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::JsonObject;
use crate::error::{NestedSequence, err};

/// The field key in the tokenizers.json file for each kind
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

    /// The key under which a `Sequence` of this kind lists its items:
    /// ```json
    /// "normalizer": {
    ///     "type": "Sequence",
    ///     "normalizers": [{ "type": "NFC" }]   // <-- the items key
    /// }
    /// ```
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

/// A component's JSON spec, as a shared reference: cloning it shares the same spec.
#[derive(Clone)]
pub(crate) enum ComponentSpec {
    Single(Arc<SingleSpec>),
    Sequence(Arc<SequenceSpec>),
}

impl ComponentSpec {
    /// Reads the `kind` component from its `tokenizer.json` object.
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

    /// An independent [`ComponentSpec`]: an edit of either one does not reach the other.
    pub(crate) fn deep_clone(&self) -> Self {
        match self {
            Self::Single(single) => Self::Single(Arc::new(single.deep_clone())),
            Self::Sequence(sequence) => Self::Sequence(Arc::new(sequence.deep_clone())),
        }
    }

    /// The component's `tokenizer.json` object.
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

/// The `SingleSpec` a `Sequence` item must be: a `Sequence` refuses another as an item.
impl TryFrom<ComponentSpec> for Arc<SingleSpec> {
    type Error = NestedSequence;

    fn try_from(spec: ComponentSpec) -> Result<Self, NestedSequence> {
        match spec {
            ComponentSpec::Single(single) => Ok(single),
            ComponentSpec::Sequence(_) => Err(NestedSequence),
        }
    }
}

/// A component that is a single JSON object: anything but a `Sequence`.
pub(crate) struct SingleSpec {
    json: Mutex<Arc<JsonObject>>,
}

impl SingleSpec {
    pub(crate) fn new(json: JsonObject) -> Self {
        Self {
            json: Mutex::new(Arc::new(json)),
        }
    }

    /// A new [`SingleSpec`] sharing the JSON until either of them is edited.
    fn deep_clone(&self) -> Self {
        Self {
            json: Mutex::new(self.json()),
        }
    }

    pub(crate) fn json(&self) -> Arc<JsonObject> {
        self.json
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Runs `edit` on the JSON. A [`deep_clone`](Self::deep_clone) sharing that JSON keeps the previous value.
    fn edit<R>(&self, edit: impl FnOnce(&mut JsonObject) -> R) -> R {
        edit(Arc::make_mut(
            &mut self.json.lock().unwrap_or_else(PoisonError::into_inner),
        ))
    }

    /// Reads the `key` field. A missing key reads as `null`, so `Option<T>` reads it as `None`.
    pub(crate) fn field<T: DeserializeOwned>(&self, key: &str) -> PyResult<T> {
        let json = self.json();
        let value = json.get(key).unwrap_or(&Value::Null);
        T::deserialize(value).map_err(|e| err(format!("`{key}`: {e}")))
    }

    pub(crate) fn set_field(&self, key: &str, value: impl Into<Value>) {
        self.edit(|json| json.insert(key.to_owned(), value.into()));
    }

    /// The `"type"` of the component, such as `"Prepend"`.
    pub(crate) fn json_type(&self) -> Option<String> {
        self.json().get("type")?.as_str().map(str::to_owned)
    }
}

/// A `Sequence` of components, each a [`SingleSpec`].
pub(crate) struct SequenceSpec {
    /// See [`ComponentKey::sequence_items_key`].
    items_key: &'static str,
    items: Mutex<Vec<Arc<SingleSpec>>>,
}

impl SequenceSpec {
    pub(crate) fn new(
        kind: ComponentKey,
        items: Vec<ComponentSpec>,
    ) -> Result<Self, NestedSequence> {
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

    /// `index` as Python reads it: a negative index counts from the end.
    pub(crate) fn index(&self, index: isize) -> Option<usize> {
        let len = self.len();
        let index = match usize::try_from(index) {
            Ok(index) => index,
            Err(_) => len.checked_sub(index.unsigned_abs())?,
        };
        (index < len).then_some(index)
    }

    /// A new [`SequenceSpec`] holding a [`SingleSpec::deep_clone`] of each item.
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

    /// The `tokenizer.json` object: the items' JSON listed under `items_key`.
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
