use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use serde_json::{Map, Value};

use tk::pipeline::PipelineTokenizer;
use tk::{
    PaddingParams, Result, TokenizerJson, TruncationParams, canonicalize_value, from_json,
    padding_from_json, padding_to_json, truncation_from_json, truncation_to_json,
};

pub(crate) type JsonObject = Map<String, Value>;

/// Holds the specification (json) of a [`PipelineTokenizer`] and compiled [`PipelineTokenizer`]
pub(crate) struct TokenizerSpec {
    /// The tokenizers.json version field
    version: String,
    /// Truncation params. Can be changed without re-compiling the [`PipelineTokenizer`].
    pub(crate) truncation: Option<TruncationParams>,
    /// Padding params. Can be changed without re-compiling the [`PipelineTokenizer`].
    pub(crate) padding: Option<PaddingParams>,
    /// Data required to build a [`PipelineTokenizer`], from the `tokenizers.json`.
    /// Invalidates [`self.compiled`] when it changes.
    pipeline_spec: PipelineSpec<Arc<ComponentSpec>>,
    /// A compiled [`PipelineTokenizer`] along with the JSON used to build it.
    /// Gets invalidated when pipeline_spec changes, and recompiled lazily
    compiled: Option<(PipelineSpecSnapshot, PipelineTokenizer)>,
}

impl TokenizerSpec {
    /// Reads a tokenizers.json
    pub(crate) fn from_json(text: &str) -> Result<Self> {
        let file = TokenizerJson::from_value(serde_json::from_str(text)?)?;
        Ok(Self {
            version: file.version,
            truncation: truncation_from_json(&file.truncation.to_string())?,
            padding: padding_from_json(&file.padding.to_string())?,
            pipeline_spec: PipelineSpec {
                role_to_token: file.role_to_token.map(Arc::new),
                added_tokens: Arc::new(file.added_tokens),
                normalizer: file.normalizer.map(ComponentSpec::new),
                pre_tokenizer: file.pre_tokenizer.map(ComponentSpec::new),
                post_processor: file.post_processor.map(ComponentSpec::new),
                decoder: file.decoder.map(ComponentSpec::new),
                model: ComponentSpec::new(file.model),
            },
            compiled: None,
        })
    }

    /// A snapshot of the current PipelineSpec
    fn take_snapshot(&self) -> PipelineSpecSnapshot {
        self.pipeline_spec.map(|spec| spec.json())
    }

    /// Gets a reference to the [`PipelineTokenizer`] built from [`self.pipeline`], compiling it if needed
    /// Changing the pipeline spec re-compiles the [`PipelineTokenizer`]
    pub(crate) fn compile(&mut self) -> Result<PipelineTokenizer> {
        // The check and the build use the same snapshot, so the cache always matches what it built.
        let snapshot = self.take_snapshot();
        if let Some((previous_snapshot, pipeline)) = &self.compiled
            && *previous_snapshot == snapshot
        {
            return Ok(pipeline.clone());
        }
        let mut tokenizer_json = Value::Object(
            snapshot
                .tokenizer_json(self.version.clone(), Value::Null, Value::Null)
                .into_object(),
        );
        canonicalize_value(&mut tokenizer_json)?;
        // XXX: the .to_string() copies JSON data a second time, could be improved
        let pipeline = from_json(&tokenizer_json.to_string())?;
        self.compiled = Some((snapshot, pipeline.clone()));
        Ok(pipeline)
    }

    /// An independent copy: editing components of the new object do not affect `self`'s components
    pub(crate) fn copy(&self) -> Self {
        Self {
            version: self.version.clone(),
            truncation: self.truncation.clone(),
            padding: self.padding.clone(),
            pipeline_spec: self.pipeline_spec.map(|spec| spec.detached()),
            compiled: self.compiled.clone(),
        }
    }

    pub(crate) fn component(&self, key: ComponentKey) -> Option<&Arc<ComponentSpec>> {
        let pipeline = &self.pipeline_spec;
        match key {
            ComponentKey::Normalizer => pipeline.normalizer.as_ref(),
            ComponentKey::PreTokenizer => pipeline.pre_tokenizer.as_ref(),
            ComponentKey::PostProcessor => pipeline.post_processor.as_ref(),
            ComponentKey::Decoder => pipeline.decoder.as_ref(),
            ComponentKey::Model => Some(&pipeline.model),
        }
    }

    pub(crate) fn set_component(&mut self, key: ComponentKey, spec: Option<Arc<ComponentSpec>>) {
        let pipeline = &mut self.pipeline_spec;
        match key {
            ComponentKey::Normalizer => pipeline.normalizer = spec,
            ComponentKey::PreTokenizer => pipeline.pre_tokenizer = spec,
            ComponentKey::PostProcessor => pipeline.post_processor = spec,
            ComponentKey::Decoder => pipeline.decoder = spec,
            ComponentKey::Model => {
                pipeline.model = spec.expect("the model setter takes a Model, never None")
            }
        }
    }

    /// Returns a snapshot of the spec as a `JsonObject`
    pub(crate) fn to_object(&self) -> JsonObject {
        self.take_snapshot()
            .tokenizer_json(
                self.version.clone(),
                params(truncation_to_json(self.truncation.as_ref())),
                params(padding_to_json(self.padding.as_ref())),
            )
            .into_object()
    }
}

#[derive(Clone, Copy)]
pub(crate) enum ComponentKey {
    Normalizer,
    PreTokenizer,
    PostProcessor,
    Decoder,
    Model,
}

/// Specification for building a [`PipelineTokenizer`].
#[derive(Clone, PartialEq)]
struct PipelineSpec<C> {
    /// Only in "2.0" files
    role_to_token: Option<Arc<Value>>,
    added_tokens: Arc<Value>,

    normalizer: Option<C>,
    pre_tokenizer: Option<C>,
    post_processor: Option<C>,
    decoder: Option<C>,
    model: C,
}

/// A snapshot of [`PipelineSpec`]
type PipelineSpecSnapshot = PipelineSpec<Arc<JsonObject>>;

impl<C> PipelineSpec<C> {
    fn map<D>(&self, f: impl Fn(&C) -> D) -> PipelineSpec<D> {
        PipelineSpec {
            role_to_token: self.role_to_token.clone(),
            added_tokens: self.added_tokens.clone(),
            normalizer: self.normalizer.as_ref().map(&f),
            pre_tokenizer: self.pre_tokenizer.as_ref().map(&f),
            post_processor: self.post_processor.as_ref().map(&f),
            decoder: self.decoder.as_ref().map(&f),
            model: f(&self.model),
        }
    }
}

impl PipelineSpecSnapshot {
    /// Clones the *content* of the snapshot as a [`TokenizerJson`]
    fn tokenizer_json(&self, version: String, truncation: Value, padding: Value) -> TokenizerJson {
        let clone_component = |object: &Option<Arc<JsonObject>>| object.as_deref().cloned();
        TokenizerJson {
            version,
            truncation,
            padding,
            role_to_token: self.role_to_token.as_deref().cloned(),
            added_tokens: Value::clone(&self.added_tokens),
            normalizer: clone_component(&self.normalizer),
            pre_tokenizer: clone_component(&self.pre_tokenizer),
            post_processor: clone_component(&self.post_processor),
            decoder: clone_component(&self.decoder),
            model: JsonObject::clone(&self.model),
        }
    }
}

/// A component's JSON spec, as a shared reference.
pub(crate) struct ComponentSpec(Mutex<Arc<JsonObject>>);

impl ComponentSpec {
    pub(crate) fn new(json: JsonObject) -> Arc<Self> {
        Arc::new(Self(Mutex::new(Arc::new(json))))
    }

    /// A new [`ComponentSpec`] over the same JSON. An edit of either one copies the JSON first.
    pub(crate) fn detached(&self) -> Arc<Self> {
        Arc::new(Self(Mutex::new(self.json())))
    }

    pub(crate) fn json(&self) -> Arc<JsonObject> {
        self.lock().clone()
    }

    /// Edits the JSON. A compiled tokenizer still holding the previous JSON keeps it.
    #[expect(dead_code, reason = "JsonFields::set_field uses it")]
    pub(crate) fn edit<R>(&self, edit: impl FnOnce(&mut JsonObject) -> R) -> R {
        edit(Arc::make_mut(&mut self.lock()))
    }

    fn lock(&self) -> MutexGuard<'_, Arc<JsonObject>> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

fn params(json: String) -> Value {
    serde_json::from_str(&json).expect("tk-serialize writes valid JSON")
}
