//! The opt-in `jinja_render_segments` v1 input schema. This is an encoding input, not a tokenizer config.
use crate::json::Json;
use tk_encode::{Result, pipeline::EncodeSegment};

/// Parse an explicit sequence of ordinary text and trusted special-token spellings.
/// Rejects unknown/duplicate/missing fields, invalid types, and unsupported versions.
/// Registration of special tokens is checked by `PipelineTokenizer::encode_segments`.
/// Producers must set `special: "yes"` only for trusted template content.
pub fn segments_from_json(text: &str) -> Result<Vec<EncodeSegment>> {
    let doc = Json::parse(text)?;
    fields(&doc, &["version", "type", "segments"])?;
    if doc.get("version").and_then(Json::as_u32_integer) != Some(1) {
        return Err("structured input requires version 1".into());
    }
    if doc.get("type").and_then(Json::as_str) != Some("jinja_render_segments") {
        return Err("structured input requires type jinja_render_segments".into());
    }
    let segments = doc.need("structured input", "segments", Json::as_array)?;
    segments
        .iter()
        .map(|segment| {
            fields(segment, &["content", "special"])?;
            let content = segment.need("segment", "content", Json::as_str)?.to_owned();
            let special = match segment.need("segment", "special", Json::as_str)? {
                "yes" => true,
                "no" => false,
                _ => return Err("segment special must be yes or no".into()),
            };
            Ok(EncodeSegment { content, special })
        })
        .collect()
}

fn fields(doc: &Json<'_>, required: &[&str]) -> Result<()> {
    let entries = doc
        .entries()
        .ok_or("structured input and segments must be JSON objects")?;
    let mut seen = vec![false; required.len()];
    for (key, _) in entries {
        let index = required
            .iter()
            .position(|name| *name == key)
            .ok_or_else(|| format!("unknown structured input field: {key}"))?;
        if seen[index] {
            return Err(format!("duplicate structured input field: {key}").into());
        }
        seen[index] = true;
    }
    if let Some(index) = seen.iter().position(|present| !present) {
        return Err(format!("missing structured input field: {}", required[index]).into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structured_json_encodes_with_a_loaded_rc0_pipeline() {
        let tokenizer = crate::from_json(
            r#"{
            "version":"2.0", "model":{"type":"BPE", "byte_level":false,
            "vocab":{"a":0,"b":1,"ab":2},"merges":[["a","b"]]},
            "added_tokens":[{"id":3,"content":"<s>","special":true,"normalized":false,
            "single_word":false,"lstrip":false,"rstrip":false}],
            "normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null
        }"#,
        )
        .unwrap();
        let segments = segments_from_json(
            r#"{"version":1,"type":"jinja_render_segments",
            "segments":[{"content":"a","special":"no"},{"content":"b<s>","special":"no"},
            {"content":"<s>","special":"yes"}]}"#,
        )
        .unwrap();
        let encodings = tokenizer.encode_segments(&segments, false).wait().unwrap();
        assert_eq!(encodings[0].ids(), &[2, 3]);
        let unknown = segments_from_json(
            r#"{"version":1,"type":"jinja_render_segments",
            "segments":[{"content":"<missing>","special":"yes"}]}"#,
        )
        .unwrap();
        assert!(tokenizer.encode_segments(&unknown, false).wait().is_err());
    }

    #[test]
    fn structured_json_preserves_unicode_and_empty_segments() {
        let segments = segments_from_json(r#"{"version":1,"type":"jinja_render_segments","segments":[{"content":"你好\n","special":"no"},{"content":"<s>","special":"yes"},{"content":"","special":"no"}]}"#).unwrap();
        assert_eq!(
            segments,
            vec![
                EncodeSegment::text("你好\n"),
                EncodeSegment::special("<s>"),
                EncodeSegment::text("")
            ]
        );
    }

    #[test]
    fn structured_json_rejects_malformed_schemas() {
        let valid = r#"{"version":1,"type":"jinja_render_segments","segments":[{"content":"x","special":"no"}]}"#;
        let invalid = vec![
            "null".to_owned(),
            "[]".to_owned(),
            "{}".to_owned(),
            valid.replace("\"version\":1", "\"version\":2"),
            valid.replace("\"version\":1", "\"version\":true"),
            valid.replace("\"version\":1", "\"version\":\"1\""),
            valid.replace("\"version\":1", "\"version\":1,\"version\":1"),
            valid.replace("jinja_render_segments", "other"),
            valid.replace("\"version\":1", "\"version\":1.0"),
            valid.replace("\"version\":1", "\"version\":1e0"),
            valid.replace("\"version\":1", "\"version\":1.00000000000000001"),
            valid.replace("\"content\":\"x\"", "\"content\":null"),
            valid.replace("\"content\":\"x\"", "\"content\":\"x\",\"content\":\"y\""),
            valid.replace("\"special\":\"no\"", "\"special\":false"),
            valid.replace("\"special\":\"no\"", "\"special\":\"maybe\""),
            valid.replace("\"special\":\"no\"", "\"extra\":\"no\""),
            valid.replace("\"version\":1", "\"pair\":null,\"version\":1"),
            format!("{valid} trailing"),
        ];
        for input in invalid {
            assert!(segments_from_json(&input).is_err(), "accepted {input}");
        }
        assert!(
            segments_from_json(r#"{"version":1,"type":"jinja_render_segments","segments":[]}"#)
                .unwrap()
                .is_empty()
        );
    }
}
