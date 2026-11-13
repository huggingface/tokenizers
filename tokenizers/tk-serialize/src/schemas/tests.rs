//! Each test writes a component as its list entry says, loads it with the reader, writes it back
//! with the writer, and compares. The reader and the writer are written by hand, apart from the
//! lists, so a key or a default can drift between them.

use serde_json::{Value, json};
use tk_encode::tokenizer::Result;

use super::Choice;
use crate::from_json;
use crate::to_json::to_json;

/// One entry of a list, such as `Strip { strip_left: Bool = true, strip_right: Bool = true }`.
struct Entry {
    /// The `tokenizer.json` key that holds the component, such as `"normalizer"`.
    json_key: &'static str,
    json_type: &'static str,
    fields: Vec<Field>,
}

struct Field {
    key: &'static str,
    default: Option<Value>,
    /// Valid values of the field's type.
    samples: Vec<Value>,
}

/// Builds the [`Entry`]s of a list.
macro_rules! entries {
    ({ $json_key:literal } $(
        $(#[doc = $doc:literal])+
        $json_type:ident $({
            $($key:ident: $field_type:ident $(($choices:expr))? $(= $default:tt)?),* $(,)?
        })?
    ),* $(,)?) => {
        vec![$(Entry {
            json_key: $json_key,
            json_type: stringify!($json_type),
            fields: vec![$($(Field {
                key: stringify!($key),
                default: default_value!($field_type $(($choices))? $(= $default)?),
                samples: sample_values!($field_type $(($choices))?),
            }),*)?],
        }),*]
    };
}

macro_rules! default_value {
    (OptionalBool = None) => {
        Some(Value::Null)
    };
    (Choice($choices:expr) = $default:tt) => {
        Some(choice_json($choices, $default))
    };
    (Char = $default:tt) => {
        Some(Value::from($default.to_string()))
    };
    ($field_type:ident = $default:tt) => {
        Some(Value::from($default))
    };
    ($field_type:ident $(($choices:expr))?) => {
        None
    };
}

macro_rules! sample_values {
    (Bool) => {
        vec![json!(false), json!(true)]
    };
    (OptionalBool) => {
        vec![json!(null), json!(false), json!(true)]
    };
    (Usize) => {
        vec![json!(1), json!(2)]
    };
    (String) => {
        vec![json!("a"), json!("b")]
    };
    (Char) => {
        vec![json!("a"), json!("b")]
    };
    // Every choice, so a misspelled `json` string fails to load.
    (Choice($choices:expr)) => {{
        let mut values = Vec::new();
        for choice in $choices {
            values.push(json!(choice.json));
        }
        values
    }};
    (Pattern) => {
        vec![json!({"String": "a"}), json!({"Regex": "b"})]
    };
    // A charsmap with an empty trie and no normalized text.
    (Base64Bytes) => {
        vec![json!("AAAAAA==")]
    };
}

// Test for mis-spelled keys. Eg, the reader reads the `individual_digits` key of a `Digits`
// pre-tokenizer. If `pre_tokenizer_schemas!` spells it `individual_digit`, the Python class
// writes that key, the reader ignores it, and `Digits(individual_digit=True)` does nothing.
#[test]
fn listed_values_round_trip() {
    let mut failures = Vec::new();
    for entry in all_entries() {
        let mut components = vec![entry.with_defaults()];
        for field in &entry.fields {
            for sample in &field.samples {
                let mut component = entry.with_defaults();
                component[field.key] = sample.clone();
                components.push(component);
            }
        }
        for component in components {
            match round_trip(entry.json_key, &component) {
                Ok(written) if written == component => {}
                Ok(written) => failures.push(format!("{component} comes back as {written}")),
                Err(e) => failures.push(format!("{component}: {e}")),
            }
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

// Test that the listed default value for a field is correct. Eg, the reader reads a `Digits`
// pre-tokenizer with no `individual_digits` key as `false`. If `pre_tokenizer_schemas!` says
// `= true`, the Python getter returns `true` for that file, but the tokenizer runs with `false`.
#[test]
fn missing_fields_read_as_their_default() {
    let mut failures = Vec::new();
    for entry in all_entries() {
        for field in &entry.fields {
            if field.default.is_none() {
                continue;
            }
            let mut component = entry.with_defaults();
            component.as_object_mut().unwrap().remove(field.key);
            // The reader may refuse a missing key instead.
            let Ok(written) = round_trip(entry.json_key, &component) else {
                continue;
            };
            if written != entry.with_defaults() {
                failures.push(format!("{component} comes back as {written}"));
            }
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

fn all_entries() -> Vec<Entry> {
    let mut entries = crate::normalizer_schemas!(entries! { "normalizer" });
    entries.extend(crate::pre_tokenizer_schemas!(entries! { "pre_tokenizer" }));
    entries.extend(crate::decoder_schemas!(entries! { "decoder" }));
    entries
}

impl Entry {
    /// The component's JSON, with each field at its default, or its first sample when it has none.
    fn with_defaults(&self) -> Value {
        let mut component = json!({"type": self.json_type});
        for field in &self.fields {
            let value = field.default.as_ref().unwrap_or(&field.samples[0]);
            component[field.key] = value.clone();
        }
        component
    }
}

/// The `json` string of the choice that Python calls `name`.
fn choice_json(choices: &[Choice], name: &str) -> Value {
    for choice in choices {
        if choice.name == name {
            return json!(choice.json);
        }
    }
    panic!("the default {name} is not one of the choices");
}

/// Loads a tokenizer holding `component` under `json_key`, writes it back, and returns the
/// component as written.
fn round_trip(json_key: &str, component: &Value) -> Result<Value> {
    let mut file = json!({
        "version": "2.0",
        "added_tokens": [],
        "model": {"type": "BPE", "byte_level": false, "vocab": {"a": 0}, "merges": []},
    });
    file[json_key] = component.clone();
    let tokenizer = from_json(&file.to_string())?;
    let mut written: Value = serde_json::from_str(&to_json(&tokenizer)?).unwrap();
    Ok(written[json_key].take())
}
