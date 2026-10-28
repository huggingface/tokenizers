//! Utils to render Tokenizers.__repr__

use serde_json::{Map, Value};
use tk::{PaddingParams, TruncationParams};

use crate::options::{Padding, Truncation};

/// How many items of a list are shown before `...`. A dict shows every field.
const MAX_PROPERTIES_SHOWN: usize = 5;

const MAX_LINE_WIDTH: usize = 100;

const INDENT: &str = "    ";

/// The value under one top-level key of a `tokenizer.json`.
pub(crate) enum Field<'a> {
    Json(&'a Value),
    Component(Option<&'a Map<String, Value>>),
    Padding(Option<&'a PaddingParams>),
    Truncation(Option<&'a TruncationParams>),
}

/// Renders a `tokenizer.json` as `name(...)`, one line per key.
pub(crate) fn tokenizer<'a>(
    name: &str,
    fields: impl IntoIterator<Item = (&'a str, Field<'a>)>,
) -> String {
    let mut out = format!("{name}(");
    for (key, field) in fields {
        newline(1, &mut out);
        out.push_str(key);
        out.push('=');
        match field {
            Field::Json(value) => write_value(key, value, Some(1), &mut out),
            Field::Component(Some(fields)) => write_object(key, fields, Some(1), &mut out),
            Field::Component(None) | Field::Padding(None) | Field::Truncation(None) => {
                out.push_str("None")
            }
            Field::Padding(Some(padding)) => {
                push_indented(&Padding::from(padding.clone()).__repr__(), &mut out)
            }
            Field::Truncation(Some(truncation)) => {
                push_indented(&Truncation::from(truncation.clone()).__repr__(), &mut out)
            }
        }
        out.push(',');
    }
    newline(0, &mut out);
    out.push(')');
    out
}

/// `key` is the one `value` sits under, `""` for a list item. `indent` is `None` while a
/// container above tries to fit on one line, so nothing spreads.
fn write_value(key: &str, value: &Value, indent: Option<usize>, out: &mut String) {
    match value {
        Value::Null => out.push_str("None"),
        Value::Bool(b) => out.push_str(if *b { "True" } else { "False" }),
        Value::Number(n) => out.push_str(&n.to_string()),
        // A charsmap is a base64 blob of hundreds of KB.
        Value::String(_) if key == "precompiled_charsmap" => out.push_str("..."),
        Value::String(s) => write_str(s, out),
        Value::Array(items) => {
            out.push('[');
            write_entries(
                items.iter().map(|item| ("", item)),
                |_, _| {},
                MAX_PROPERTIES_SHOWN,
                indent,
                out,
            );
            out.push(']');
        }
        Value::Object(fields) => write_object(key, fields, indent, out),
    }
}

fn write_object(key: &str, fields: &Map<String, Value>, indent: Option<usize>, out: &mut String) {
    if let Some(Value::String(tag)) = fields.get("type") {
        out.push_str(tag);
        out.push('(');
        write_entries(
            fields
                .iter()
                .filter(|(key, _)| *key != "type")
                .map(|(key, value)| (key.as_str(), value)),
            |key, out| {
                out.push_str(key);
                out.push('=');
            },
            usize::MAX,
            indent,
            out,
        );
        out.push(')');
    } else {
        // Truncate vocab to 5 entries
        let limit = if key == "vocab" {
            MAX_PROPERTIES_SHOWN
        } else {
            usize::MAX
        };
        out.push('{');
        write_entries(
            fields.iter().map(|(key, value)| (key.as_str(), value)),
            |key, out| {
                write_str(key, out);
                out.push_str(": ");
            },
            limit,
            indent,
            out,
        );
        out.push('}');
    }
}

/// The entries stay on one line when it fits in [`MAX_LINE_WIDTH`] columns.
/// Otherwise one per line.
fn write_entries<'a>(
    entries: impl Iterator<Item = (&'a str, &'a Value)>,
    write_key: fn(&str, &mut String),
    limit: usize,
    indent: Option<usize>,
    out: &mut String,
) {
    let entries: Vec<_> = entries.take(limit.saturating_add(1)).collect();
    let line = out.rfind('\n').map_or(0, |i| i + 1);
    let start = out.len();
    write_entries_at(&entries, write_key, limit, None, out);
    let Some(indent) = indent else { return };
    if out[line..].chars().count() + 2 <= MAX_LINE_WIDTH {
        return;
    }
    out.truncate(start);
    write_entries_at(&entries, write_key, limit, Some(indent + 1), out);
    newline(indent, out);
}

fn write_entries_at(
    entries: &[(&str, &Value)],
    write_key: fn(&str, &mut String),
    limit: usize,
    indent: Option<usize>,
    out: &mut String,
) {
    for (i, (key, value)) in entries.iter().enumerate() {
        match indent {
            Some(indent) => newline(indent, out),
            None if i > 0 => out.push_str(", "),
            None => {}
        }
        if i == limit {
            out.push_str("...");
        } else {
            write_key(key, out);
            write_value(key, value, indent, out);
        }
        if indent.is_some() {
            out.push(',');
        }
    }
}

fn push_indented(rendered: &str, out: &mut String) {
    out.push_str(&rendered.replace('\n', &format!("\n{INDENT}")));
}

fn newline(indent: usize, out: &mut String) {
    out.push('\n');
    out.push_str(&INDENT.repeat(indent));
}

fn write_str(s: &str, out: &mut String) {
    out.push_str(&tk::str_to_json(s));
}
