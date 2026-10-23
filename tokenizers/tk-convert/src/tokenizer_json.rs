use serde_json::{Map, Value};

use crate::convert::{ConvertError, fill_model_type, kind_of};

type JsonObject = Map<String, Value>;

const VERSIONS: [&str; 2] = ["1.0", "2.0"];

const KEYS: [&str; 9] = [
    "version",
    "truncation",
    "padding",
    "added_tokens",
    "normalizer",
    "pre_tokenizer",
    "post_processor",
    "decoder",
    "model",
];

/// A `tokenizer.json` split into its top-level keys.
///
/// Each value is the JSON the file holds. [`TokenizerJson::from_value`] only checks the file's
/// shape and fills in a missing model `"type"`.
///
/// [`canonicalize_value`](crate::canonicalize_value) rewrites the rest.
#[derive(Clone, Debug, PartialEq)]
pub struct TokenizerJson {
    /// Default to `"1.0"` when the file omits the field
    pub version: String,
    pub truncation: Value,
    pub padding: Value,
    /// Only present in `2.0` files.
    pub role_to_token: Option<Value>,
    pub added_tokens: Value,
    pub normalizer: Option<JsonObject>,
    pub pre_tokenizer: Option<JsonObject>,
    pub post_processor: Option<JsonObject>,
    pub decoder: Option<JsonObject>,
    pub model: JsonObject,
}

impl TokenizerJson {
    pub fn from_value(value: Value) -> Result<Self, ConvertError> {
        let Value::Object(mut file) = value else {
            let found = kind_of(&value);
            return Err(ConvertError::NotAnObject { found });
        };
        let version = match file.get("version") {
            // If the file has no "version" field, default to "1.0"
            None => "1.0".to_owned(),
            Some(Value::String(v)) if VERSIONS.contains(&v.as_str()) => v.clone(),
            Some(v) => {
                return Err(ConvertError::UnknownVersion {
                    version: v.to_string(),
                });
            }
        };
        // Scanned before any removal: with `preserve_order`, `Map::remove` reorders the keys.
        let unknown_keys: Vec<String> = file
            .keys()
            .filter(|key| {
                !KEYS.contains(&key.as_str())
                    && !(key.as_str() == "role_to_token" && version.as_str() == "2.0")
            })
            .cloned()
            .collect();
        if !unknown_keys.is_empty() {
            return Err(ConvertError::UnknownKeys { keys: unknown_keys });
        }
        let mut model = match file.remove("model") {
            Some(Value::Object(model)) => model,
            Some(other) => {
                return Err(ConvertError::ModelNotObject {
                    found: kind_of(&other),
                });
            }
            None => return Err(ConvertError::MissingModel),
        };
        // Older files leave the model's "type" out
        fill_model_type(&mut model);
        Ok(Self {
            version,
            truncation: file.remove("truncation").unwrap_or(Value::Null),
            padding: file.remove("padding").unwrap_or(Value::Null),
            role_to_token: file.remove("role_to_token"),
            added_tokens: file.remove("added_tokens").unwrap_or(Value::Null),
            normalizer: take_component(&mut file, "normalizer")?,
            pre_tokenizer: take_component(&mut file, "pre_tokenizer")?,
            post_processor: take_component(&mut file, "post_processor")?,
            decoder: take_component(&mut file, "decoder")?,
            model,
        })
    }

    /// Convert to JsonObject, respect 0.23 key order
    pub fn into_object(self) -> JsonObject {
        let component = |object: Option<JsonObject>| object.map_or(Value::Null, Value::Object);
        let head = [
            ("version", Value::String(self.version)),
            ("truncation", self.truncation),
            ("padding", self.padding),
        ];
        let role_to_token = self.role_to_token.map(|roles| ("role_to_token", roles));
        let tail = [
            ("added_tokens", self.added_tokens),
            ("normalizer", component(self.normalizer)),
            ("pre_tokenizer", component(self.pre_tokenizer)),
            ("post_processor", component(self.post_processor)),
            ("decoder", component(self.decoder)),
            ("model", Value::Object(self.model)),
        ];
        head.into_iter()
            .chain(role_to_token)
            .chain(tail)
            .map(|(key, value)| (key.to_owned(), value))
            .collect()
    }
}

fn take_component(
    file: &mut JsonObject,
    key: &'static str,
) -> Result<Option<JsonObject>, ConvertError> {
    match file.remove(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Object(object)) => Ok(Some(object)),
        Some(other) => Err(ConvertError::ComponentNotObject {
            key,
            found: kind_of(&other),
        }),
    }
}
