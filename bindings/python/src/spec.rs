use std::sync::Arc;

use serde_json::{Map, Value};

use tk::pipeline::PipelineTokenizer;
use tk::{
    PaddingParams, Result, TruncationParams, canonicalize_value, fill_model_type, from_json,
    padding_from_json, padding_to_json, truncation_from_json, truncation_to_json,
};

pub(crate) type JsonObject = Map<String, Value>;

const VERSIONS: [&str; 2] = ["1.0", "2.0"];

/// Holds the specification (json) of a [`PipelineTokenizer`], and the [`PipelineTokenizer`] last
/// compiled from it. Compilation is lazy.
#[derive(Clone)]
pub(crate) struct TokenizerSpec {
    /// The tokenizers.json version field
    version: String,
    /// Truncation params. Can be changed without re-compiling the [`PipelineTokenizer`].
    pub(crate) truncation: Option<TruncationParams>,
    /// Padding params. Can be changed without re-compiling the [`PipelineTokenizer`].
    pub(crate) padding: Option<PaddingParams>,
    /// Data required to build a [`PipelineTokenizer`], from the tokenizers.json.
    /// Invalidates [`self.compiled`] when it changes.
    pipeline: PipelineSpec,
    /// A compiled [`PipelineTokenizer`] along with the spec used to build it
    /// When the spec changes, it gets re-compiled
    compiled: Option<(PipelineSpec, PipelineTokenizer)>,
}

/// Required data to build a [`PipelineTokenizer`]
#[derive(Clone, PartialEq)]
struct PipelineSpec {
    added_tokens: Arc<Value>,
    normalizer: Option<Arc<JsonObject>>,
    pre_tokenizer: Option<Arc<JsonObject>>,
    post_processor: Option<Arc<JsonObject>>,
    decoder: Option<Arc<JsonObject>>,
    model: Arc<JsonObject>,
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
        let mut model = take_object(&mut file, "model")?;
        // Older files leave the model's "type" out. 0.23 fills it in when it saves the file.
        fill_model_type(&mut model);
        let spec = Self {
            version,
            truncation: truncation_from_json(&take(&mut file, "truncation").to_string())?,
            padding: padding_from_json(&take(&mut file, "padding").to_string())?,
            pipeline: PipelineSpec {
                added_tokens: Arc::new(take(&mut file, "added_tokens")),
                normalizer: take_optional_object(&mut file, "normalizer")?,
                pre_tokenizer: take_optional_object(&mut file, "pre_tokenizer")?,
                post_processor: take_optional_object(&mut file, "post_processor")?,
                decoder: take_optional_object(&mut file, "decoder")?,
                model: Arc::new(model),
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
        if let Some((spec, pipeline)) = &self.compiled
            && *spec == self.pipeline
        {
            return Ok(pipeline.clone());
        }
        let mut file = Value::Object(self.to_object());
        canonicalize_value(&mut file)?;
        let pipeline = from_json(&file.to_string())?;
        self.compiled = Some((self.pipeline.clone(), pipeline.clone()));
        Ok(pipeline)
    }

    pub(crate) fn to_object(&self) -> JsonObject {
        let spec = &self.pipeline;
        let keys = [
            ("version", Value::from(self.version.as_str())),
            (
                "truncation",
                params(truncation_to_json(self.truncation.as_ref())),
            ),
            ("padding", params(padding_to_json(self.padding.as_ref()))),
            ("added_tokens", Value::clone(&spec.added_tokens)),
            ("normalizer", optional_object(&spec.normalizer)),
            ("pre_tokenizer", optional_object(&spec.pre_tokenizer)),
            ("post_processor", optional_object(&spec.post_processor)),
            ("decoder", optional_object(&spec.decoder)),
            ("model", Value::Object(JsonObject::clone(&spec.model))),
        ];
        keys.into_iter().map(|(k, v)| (k.to_owned(), v)).collect()
    }

    pub(crate) fn to_json(&self) -> String {
        Value::Object(self.to_object()).to_string()
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

fn take_optional_object(file: &mut JsonObject, key: &str) -> Result<Option<Arc<JsonObject>>> {
    match take(file, key) {
        Value::Null => Ok(None),
        Value::Object(object) => Ok(Some(Arc::new(object))),
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
