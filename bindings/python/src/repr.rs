//! Utils to render Tokenizers.__repr__

use std::fmt::Write;

use serde_json::{Map, Value};

/// The keys of a `tokenizer.json`, in the order `tk_serialize::to_json` writes them.
const KEYS: [&str; 10] = [
    "version",
    "truncation",
    "padding",
    "role_to_token",
    "added_tokens",
    "normalizer",
    "pre_tokenizer",
    "post_processor",
    "decoder",
    "model",
];

/// How many entries of a list or dict are shown before `...`. A `{"type": ...}` object is a
/// component's configuration and shows every field.
const SHOWN: usize = 5;

/// `padding` is what applies now, already rendered; the file's own `padding` block is what
/// `from_file` read and may since have been replaced.
pub(crate) fn tokenizer(file: &Map<String, Value>, padding: &str) -> String {
    let known = KEYS.into_iter().filter(|key| file.contains_key(*key));
    let extra = file
        .keys()
        .map(String::as_str)
        .filter(|key| !KEYS.contains(key));
    let mut out = String::from("Tokenizer(");
    for (i, key) in known.chain(extra).enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(key);
        out.push('=');
        if key == "padding" {
            out.push_str(padding);
        } else {
            write_value(&file[key], &mut out);
        }
    }
    out.push(')');
    out
}

fn write_value(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("None"),
        Value::Bool(true) => out.push_str("True"),
        Value::Bool(false) => out.push_str("False"),
        Value::Number(n) => write!(out, "{n}").unwrap(),
        Value::String(s) => write_str(s, out),
        Value::Array(items) => {
            out.push('[');
            write_entries(items.iter().map(|item| ("", item)), |_, _| {}, SHOWN, out);
            out.push(']');
        }
        Value::Object(fields) => match fields.get("type").and_then(Value::as_str) {
            Some(tag) => {
                out.push_str(tag);
                out.push('(');
                let fields = fields.iter().filter(|(key, _)| *key != "type");
                write_entries(
                    fields.map(|(key, value)| (key.as_str(), value)),
                    |key, out| {
                        out.push_str(key);
                        out.push('=');
                    },
                    usize::MAX,
                    out,
                );
                out.push(')');
            }
            None => {
                out.push('{');
                write_entries(
                    fields.iter().map(|(key, value)| (key.as_str(), value)),
                    |key, out| {
                        write_str(key, out);
                        out.push_str(": ");
                    },
                    SHOWN,
                    out,
                );
                out.push('}');
            }
        },
    }
}

fn write_entries<'a>(
    entries: impl Iterator<Item = (&'a str, &'a Value)>,
    write_key: fn(&str, &mut String),
    shown: usize,
    out: &mut String,
) {
    for (i, (key, value)) in entries.enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        if i == shown {
            out.push_str("...");
            return;
        }
        write_key(key, out);
        write_value(value, out);
    }
}

/// A JSON string literal is also a Python one.
fn write_str(s: &str, out: &mut String) {
    out.push_str(&serde_json::to_string(s).unwrap());
}
