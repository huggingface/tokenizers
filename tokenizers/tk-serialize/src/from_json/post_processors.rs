//! The `post_processor` object, lowered into the config path's `Template` IR.

use super::unsupported;
use crate::json::Json;
use tk_encode::pipeline::{PipelinePostProcessor, PipelineToken, Template};
use tk_encode::tokenizer::Result;

pub(super) fn read_post_processor(cfg: Option<&Json<'_>>) -> Result<PipelinePostProcessor> {
    let Some(cfg) = cfg else {
        return Ok(PipelinePostProcessor::default());
    };
    let kind = cfg
        .type_tag()
        .ok_or_else(|| unsupported("a post-processor with no `type`"))?;
    match kind {
        // The only spelling there is. `ByteLevel`, `BertProcessing`, `RobertaProcessing` and a
        // `Sequence` wrapper are legacy names that tk-convert rewrites into this one.
        "TemplateProcessing" => read_template(cfg),
        other => Err(unsupported(&format!("the `{other}` post-processor"))),
    }
}

fn read_template(cfg: &Json<'_>) -> Result<PipelinePostProcessor> {
    let template_for = |key: &str| -> Result<Template> {
        let pieces = cfg.need("TemplateProcessing", key, Json::as_array)?;
        let mut t = Template::default();
        // Specials land before A, between A and B, or after the last sequence -- the only three
        // places a piece can be.
        let (mut prefix, mut infix, mut suffix) = (Vec::new(), Vec::new(), Vec::new());
        let mut seen_a = false;
        for piece in pieces {
            // `type_id` is the sequence a piece belongs to, and `0` for all but the second half of
            // a pair, so it is written only when it is not that.
            let type_id = piece.field("type_id").and_then(Json::as_u32).unwrap_or(0) as u8;
            if let Some(seq) = piece.field("seq").and_then(Json::as_str) {
                match seq {
                    "A" if !seen_a => {
                        seen_a = true;
                        t.a_type_id = type_id;
                    }
                    "B" if seen_a && t.b_type_id.is_none() => t.b_type_id = Some(type_id),
                    "A" | "B" => {
                        return Err(format!(
                            "not supported: template references sequence {seq} out of order or more than once"
                        )
                        .into());
                    }
                    other => return Err(format!("unknown template sequence {other:?}").into()),
                }
            } else if let Some(ids) = piece.field("ids").and_then(Json::as_array) {
                let dst = match (seen_a, t.b_type_id.is_some(), key) {
                    (false, ..) => &mut prefix,
                    (true, false, "pair") => &mut infix,
                    _ => &mut suffix,
                };
                for id in ids {
                    let id = id.as_u32().ok_or_else(|| -> tk_encode::Error {
                        "a template piece has a bad id".into()
                    })?;
                    dst.push((PipelineToken::from(id), type_id));
                }
            } else {
                // The `{"SpecialToken": {"id": ...}}` wrapper and its `special_tokens` table are a
                // `1.0` spelling, and a `1.0` file never reaches this reader.
                return Err("a template piece has neither `seq` nor `ids`".into());
            }
        }
        if !seen_a {
            return Err("not supported: template does not reference sequence A".into());
        }
        if (key == "pair") != t.b_type_id.is_some() {
            return Err(
                format!("not supported: `{key}` template references the wrong sequences").into(),
            );
        }
        (t.prefix, t.infix, t.suffix) = (prefix.into(), infix.into(), suffix.into());
        Ok(t)
    };

    Ok(PipelinePostProcessor {
        single: template_for("single")?,
        pair: template_for("pair")?,
    })
}
