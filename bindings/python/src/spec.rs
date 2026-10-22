use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use serde_json::{Map, Value};

use tk::pipeline::PipelineTokenizer;
use tk::{
    PaddingParams, Result, TruncationParams, canonicalize_value, fill_model_type, from_json,
    padding_from_json, padding_to_json, truncation_from_json, truncation_to_json,
};

pub(crate) type JsonObject = Map<String, Value>;

const VERSIONS: [&str; 2] = ["1.0", "2.0"];

/// A component's JSON. Every handle on a component and every tokenizer holding it share one
/// [`ComponentSpec`], so an edit through any of them reaches all of them.
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

/// Holds the specification (json) of a [`PipelineTokenizer`], and the [`PipelineTokenizer`] last
/// compiled from it. Compilation is lazy.
pub(crate) struct TokenizerSpec {
    /// The tokenizers.json version field
    version: String,
    /// Truncation params. Can be changed without re-compiling the [`PipelineTokenizer`].
    pub(crate) truncation: Option<TruncationParams>,
    /// Padding params. Can be changed without re-compiling the [`PipelineTokenizer`].
    pub(crate) padding: Option<PaddingParams>,
    /// Data required to build a [`PipelineTokenizer`], from the tokenizers.json.
    /// Invalidates [`self.compiled`] when it changes.
    pipeline: PipelineSpec<Arc<ComponentSpec>>,
    /// A compiled [`PipelineTokenizer`] along with the JSON used to build it
    /// When the JSON changes, it gets re-compiled
    compiled: Option<(PipelineSpec<Arc<JsonObject>>, PipelineTokenizer)>,
}

#[derive(Clone, Copy)]
pub(crate) enum ComponentKey {
    Normalizer,
    PreTokenizer,
    PostProcessor,
    Decoder,
    Model,
}

/// Required data to build a [`PipelineTokenizer`]. A tokenizer holds each component as an
/// `Arc<ComponentSpec>`, shared with the handles on it. The compile cache holds `Arc<JsonObject>`s.
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

impl TokenizerSpec {
    /// Reads a tokenizers.json
    pub(crate) fn from_json(text: &str) -> Result<Self> {
        let Value::Object(mut file) = serde_json::from_str(text)? else {
            return Err("a tokenizer.json must be a JSON object".into());
        };
        let version = match file.shift_remove("version") {
            None => "1.0".to_owned(),
            Some(Value::String(v)) if VERSIONS.contains(&v.as_str()) => v,
            Some(v) => return Err(format!("unknown tokenizer version {v}").into()),
        };
        // 0.23 refuses `role_to_token`, so a "1.0" file leaves it to the unknown keys below.
        let role_to_token = match version.as_str() {
            "2.0" => file.shift_remove("role_to_token").map(Arc::new),
            _ => None,
        };
        let mut model = take_object(&mut file, "model")?;
        // Older files leave the model's "type" out. 0.23 fills it in when it saves the file.
        fill_model_type(&mut model);
        let spec = Self {
            version,
            truncation: truncation_from_json(&take(&mut file, "truncation").to_string())?,
            padding: padding_from_json(&take(&mut file, "padding").to_string())?,
            pipeline: PipelineSpec {
                role_to_token,
                added_tokens: Arc::new(take(&mut file, "added_tokens")),
                normalizer: take_optional_object(&mut file, "normalizer")?,
                pre_tokenizer: take_optional_object(&mut file, "pre_tokenizer")?,
                post_processor: take_optional_object(&mut file, "post_processor")?,
                decoder: take_optional_object(&mut file, "decoder")?,
                model: ComponentSpec::new(model),
            },
            compiled: None,
        };
        if !file.is_empty() {
            let keys: Vec<String> = file.keys().map(|key| format!("`{key}`")).collect();
            return Err(format!("unknown keys: {}", keys.join(", ")).into());
        }
        Ok(spec)
    }

    /// Gets a reference to the [`PipelineTokenizer`] built from [`self.pipeline`], compiling it if needed
    /// Changing the pipeline spec re-compiles the [`PipelineTokenizer`]
    pub(crate) fn compile(&mut self) -> Result<PipelineTokenizer> {
        // The check and the build use the same snapshot, so the cache always matches what it built.
        let json = self.pipeline.map(|spec| spec.json());
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
            pipeline: self.pipeline.map(|spec| spec.detached()),
            compiled: self.compiled.clone(),
        }
    }

    pub(crate) fn component(&self, key: ComponentKey) -> Option<&Arc<ComponentSpec>> {
        let pipeline = &self.pipeline;
        match key {
            ComponentKey::Normalizer => pipeline.normalizer.as_ref(),
            ComponentKey::PreTokenizer => pipeline.pre_tokenizer.as_ref(),
            ComponentKey::PostProcessor => pipeline.post_processor.as_ref(),
            ComponentKey::Decoder => pipeline.decoder.as_ref(),
            ComponentKey::Model => Some(&pipeline.model),
        }
    }

    pub(crate) fn set_component(&mut self, key: ComponentKey, spec: Option<Arc<ComponentSpec>>) {
        let pipeline = &mut self.pipeline;
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

    pub(crate) fn to_object(&self) -> JsonObject {
        self.file(&self.pipeline.map(|spec| spec.json()))
    }

    pub(crate) fn to_json(&self) -> String {
        Value::Object(self.to_object()).to_string()
    }
    fn file(&self, spec: &PipelineSpec<Arc<JsonObject>>) -> JsonObject {
        let head = [
            ("version", Value::from(self.version.as_str())),
            (
                "truncation",
                params(truncation_to_json(self.truncation.as_ref())),
            ),
            ("padding", params(padding_to_json(self.padding.as_ref()))),
        ];
        let role_to_token = spec
            .role_to_token
            .as_deref()
            .map(|roles| ("role_to_token", roles.clone()));
        let tail = [
            ("added_tokens", Value::clone(&spec.added_tokens)),
            ("normalizer", optional_object(&spec.normalizer)),
            ("pre_tokenizer", optional_object(&spec.pre_tokenizer)),
            ("post_processor", optional_object(&spec.post_processor)),
            ("decoder", optional_object(&spec.decoder)),
            ("model", Value::Object(JsonObject::clone(&spec.model))),
        ];
        head.into_iter()
            .chain(role_to_token)
            .chain(tail)
            .map(|(k, v)| (k.to_owned(), v))
            .collect()
    }
}

fn take(file: &mut JsonObject, key: &str) -> Value {
    file.shift_remove(key).unwrap_or(Value::Null)
}

fn take_object(file: &mut JsonObject, key: &str) -> Result<JsonObject> {
    match take(file, key) {
        Value::Object(object) => Ok(object),
        _ => Err(format!("`{key}` must be an object").into()),
    }
}

fn take_optional_object(file: &mut JsonObject, key: &str) -> Result<Option<Arc<ComponentSpec>>> {
    match take(file, key) {
        Value::Null => Ok(None),
        Value::Object(object) => Ok(Some(ComponentSpec::new(object))),
        _ => Err(format!("`{key}` must be an object or null").into()),
    }
}

fn optional_object(object: &Option<Arc<JsonObject>>) -> Value {
    object
        .as_deref()
        .map_or(Value::Null, |object| Value::Object(object.clone()))
}

fn params(json: String) -> Value {
    serde_json::from_str(&json).expect("tk-serialize writes valid JSON")
}
