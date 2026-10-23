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
    compiled: Option<(PipelineSpec<Arc<JsonObject>>, PipelineTokenizer)>,
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

    /// Gets a reference to the [`PipelineTokenizer`] built from [`self.pipeline`], compiling it if needed
    /// Changing the pipeline spec re-compiles the [`PipelineTokenizer`]
    pub(crate) fn compile(&mut self) -> Result<PipelineTokenizer> {
        // The check and the build use the same snapshot, so the cache always matches what it built.
        let json = self.pipeline_spec.map(|spec| spec.json());
        if let Some((compiled, pipeline)) = &self.compiled
            && *compiled == json
        {
            return Ok(pipeline.clone());
        }
        let mut file = Value::Object(self.file(&json));
        canonicalize_value(&mut file)?;
        let pipeline = from_json(&file.to_string())?;
        self.compiled = Some((json, pipeline.clone()));
        Ok(pipeline)
    }

    /// A copy whose components are its own: editing one leaves this tokenizer unchanged.
    pub(crate) fn copy(&self) -> Self {
        Self {
            version: self.version.clone(),
            truncation: self.truncation.clone(),
            padding: self.padding.clone(),
            pipeline_spec: self.pipeline_spec.map(|spec| spec.detached()),
            compiled: self.compiled.clone(),
        }
    }

    pub(crate) fn to_object(&self) -> JsonObject {
        self.file(&self.pipeline_spec.map(|spec| spec.json()))
    }

    pub(crate) fn to_json(&self) -> String {
        Value::Object(self.to_object()).to_string()
    }
    fn file(&self, spec: &PipelineSpec<Arc<JsonObject>>) -> JsonObject {
        let component = |object: &Option<Arc<JsonObject>>| object.as_deref().cloned();
        TokenizerJson {
            version: self.version.clone(),
            truncation: params(truncation_to_json(self.truncation.as_ref())),
            padding: params(padding_to_json(self.padding.as_ref())),
            role_to_token: spec.role_to_token.as_deref().cloned(),
            added_tokens: Value::clone(&spec.added_tokens),
            normalizer: component(&spec.normalizer),
            pre_tokenizer: component(&spec.pre_tokenizer),
            post_processor: component(&spec.post_processor),
            decoder: component(&spec.decoder),
            model: JsonObject::clone(&spec.model),
        }
        .into_object()
    }
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

    fn lock(&self) -> MutexGuard<'_, Arc<JsonObject>> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

fn params(json: String) -> Value {
    serde_json::from_str(&json).expect("tk-serialize writes valid JSON")
}
