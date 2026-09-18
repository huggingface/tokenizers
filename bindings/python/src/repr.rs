//! Utils to render Tokenizers.__repr__

use tk_serialize::json::Json;

/// How many items of a list are shown before `...`. A dict shows every field.
const MAX_PROPERTIES_SHOWN: usize = 5;

const MAX_LINE_WIDTH: usize = 100;

const INDENT: &str = "    ";

pub(crate) fn tokenizer(file: &Json<'_>, padding: &str) -> String {
    let mut out = String::from("Tokenizer(");
    for (key, value) in file.entries().unwrap() {
        newline(1, &mut out);
        out.push_str(key);
        out.push('=');
        if key == "padding" {
            out.push_str(&padding.replace('\n', &format!("\n{INDENT}")));
        } else {
            write_value(key, value, Some(1), &mut out);
        }
        out.push(',');
    }
    newline(0, &mut out);
    out.push(')');
    out
}

/// `key` is the one `value` sits under, `""` for a list item. `indent` is `None` while a
/// container above tries to fit on one line, so nothing spreads.
fn write_value(key: &str, value: &Json<'_>, indent: Option<usize>, out: &mut String) {
    if value.is_null() {
        out.push_str("None");
    } else if let Some(b) = value.as_bool() {
        out.push_str(if b { "True" } else { "False" });
    } else if let Some(n) = value.number_literal() {
        out.push_str(n);
    } else if let Some(s) = value.as_str() {
        write_str(s, out);
    } else if let Some(items) = value.as_array() {
        out.push('[');
        write_entries(
            items.iter().map(|item| ("", item)),
            |_, _| {},
            MAX_PROPERTIES_SHOWN,
            indent,
            out,
        );
        out.push(']');
    } else if let Some(tag) = value.type_tag() {
        out.push_str(tag);
        out.push('(');
        let fields = value.entries().unwrap().filter(|(key, _)| *key != "type");
        write_entries(
            fields,
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
            value.entries().unwrap(),
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
    entries: impl Iterator<Item = (&'a str, &'a Json<'a>)>,
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
    entries: &[(&str, &Json<'_>)],
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

fn newline(indent: usize, out: &mut String) {
    out.push('\n');
    out.push_str(&INDENT.repeat(indent));
}

fn write_str(s: &str, out: &mut String) {
    out.push_str(&tk_serialize::str_to_json(s));
}
