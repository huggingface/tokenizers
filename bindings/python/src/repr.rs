//! Utils to render Tokenizers.__repr__

use std::fmt::Write as _;

use serde_json::{Map, Value};
use tk::pipeline::{PipelineModel, PipelineToken, PipelineTokenizer, Template};
use tk::{PaddingParams, TruncationParams};

use crate::options::{Padding, Truncation};

/// How many items of a list are shown before `...`. A dict shows every field.
const MAX_PROPERTIES_SHOWN: usize = 5;

const MAX_LINE_WIDTH: usize = 100;

const INDENT: &str = "    ";

/// Renders `file` as `TokenizerSpec(...)`, one line per `tokenizer.json` key.
pub(crate) fn tokenizer_spec(
    file: &Map<String, Value>,
    padding: Option<&PaddingParams>,
    truncation: Option<&TruncationParams>,
) -> String {
    let mut out = String::from("TokenizerSpec(");
    for (key, value) in file {
        field(key, &mut out, |out| match key.as_str() {
            "padding" => push_indented(&padding_repr(padding), out),
            "truncation" => push_indented(&truncation_repr(truncation), out),
            _ => write_value(key, value, Some(1), out),
        });
    }
    newline(0, &mut out);
    out.push(')');
    out
}

/// Renders `CompiledTokenizer(...)` from the components the built pipeline runs, in the order it
/// runs them. A component the pipeline rebuilt as another shows under its new kind, such as a
/// Metaspace pre-tokenizer among the normalizers.
pub(crate) fn compiled_tokenizer(pipeline: &PipelineTokenizer) -> tk::Result<String> {
    let added_tokens = component(tk::added_tokens_to_json(pipeline.get_added_vocabulary()));
    let normalizer = component(tk::normalizer_to_json(pipeline.get_normalizers())?);
    let pre_tokenizer = component(tk::pre_tokenizer_to_json(pipeline.get_pre_tokenizer())?);
    let decoder = component(tk::decoder_to_json(pipeline.get_decoder())?);
    let post_processor = pipeline.get_post_processor();
    let mut out = String::from("CompiledTokenizer(");
    field("added_tokens", &mut out, |out| {
        write_value("added_tokens", &added_tokens, Some(1), out)
    });
    field("normalizer", &mut out, |out| {
        write_value("normalizer", &normalizer, Some(1), out)
    });
    field("pre_tokenizer", &mut out, |out| {
        write_value("pre_tokenizer", &pre_tokenizer, Some(1), out)
    });
    // The model's JSON is the whole vocabulary, so only its kind shows.
    field("model", &mut out, |out| {
        out.push_str(model_name(pipeline.get_model()))
    });
    field("post_processor", &mut out, |out| {
        write!(
            out,
            "Template(single={}, pair={})",
            template(pipeline, &post_processor.single),
            template(pipeline, &post_processor.pair)
        )
        .expect("writing to a String cannot fail")
    });
    field("decoder", &mut out, |out| {
        write_value("decoder", &decoder, Some(1), out)
    });
    field("padding", &mut out, |out| {
        push_indented(&padding_repr(pipeline.get_padding()), out)
    });
    field("truncation", &mut out, |out| {
        push_indented(&truncation_repr(pipeline.get_truncation()), out)
    });
    newline(0, &mut out);
    out.push(')');
    Ok(out)
}

fn component(json: String) -> Value {
    serde_json::from_str(&json).expect("tk-serialize writes valid JSON")
}

fn field(key: &str, out: &mut String, write: impl FnOnce(&mut String)) {
    newline(1, out);
    out.push_str(key);
    out.push('=');
    write(out);
    out.push(',');
}

fn model_name(model: &PipelineModel) -> &'static str {
    match model {
        PipelineModel::BPE(_) => "BPE",
        PipelineModel::Unigram(_) => "Unigram",
        PipelineModel::WordLevel(_) => "WordLevel",
        PipelineModel::WordPiece(_) => "WordPiece",
    }
}

/// Spells `template` the way a `TemplateProcessing` string does: `[CLS] $A [SEP] $B:1 [SEP]:1`.
fn template(pipeline: &PipelineTokenizer, template: &Template) -> String {
    let pieces: Vec<String> = tokens(pipeline, &template.prefix)
        .chain([("$A".to_owned(), template.a_type_id)])
        .chain(tokens(pipeline, &template.infix))
        .chain(template.b_type_id.map(|type_id| ("$B".to_owned(), type_id)))
        .chain(tokens(pipeline, &template.suffix))
        .map(|(piece, type_id)| match type_id {
            0 => piece,
            _ => format!("{piece}:{type_id}"),
        })
        .collect();
    tk::str_to_json(&pieces.join(" "))
}

fn tokens<'a>(
    pipeline: &'a PipelineTokenizer,
    run: &'a [(PipelineToken, u8)],
) -> impl Iterator<Item = (String, u8)> + 'a {
    run.iter().map(|&(token, type_id)| {
        let content = pipeline.decode_tokens(&[token.id()], false).concat();
        (content, type_id)
    })
}

fn padding_repr(padding: Option<&PaddingParams>) -> String {
    padding.cloned().map_or_else(
        || "None".to_owned(),
        |padding| Padding::from(padding).__repr__(),
    )
}

fn truncation_repr(truncation: Option<&TruncationParams>) -> String {
    truncation.cloned().map_or_else(
        || "None".to_owned(),
        |truncation| Truncation::from(truncation).__repr__(),
    )
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
