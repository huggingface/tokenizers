//! The `post_processor` object, lowered into the config path's `Template` IR.

use super::unsupported;
use crate::json::Json;
use tk_encode::pipeline::{PipelinePostProcessor, PipelineToken, Template, compose};
use tk_encode::tokenizer::Result;

pub(super) fn read_post_processor(cfg: Option<&Json<'_>>) -> Result<PipelinePostProcessor> {
    let Some(cfg) = cfg else {
        return Ok(PipelinePostProcessor::default());
    };
    let kind = cfg
        .type_tag()
        .ok_or_else(|| unsupported("a post-processor with no `type`"))?;
    match kind {
        "ByteLevel" => Ok(PipelinePostProcessor::default()),
        "Sequence" => {
            let members = cfg
                .field("processors")
                .and_then(Json::as_array)
                .unwrap_or(&[]);
            let built = members
                .iter()
                .map(|m| read_post_processor(Some(m)))
                .collect::<Result<Vec<_>>>()?;
            Ok(PipelinePostProcessor {
                single: compose(built.iter().map(|m| &m.single))?,
                pair: compose(built.iter().map(|m| &m.pair))?,
            })
        }
        "TemplateProcessing" => read_template(cfg),
        // type id at 0, where bert retags the second sequence.
        "BertProcessing" | "RobertaProcessing" => {
            let cls = read_special_id(cfg, "cls")?;
            let sep = read_special_id(cfg, "sep")?;
            let run = |ids: &[u32], type_id: u8| -> Box<[(PipelineToken, u8)]> {
                ids.iter()
                    .map(|&id| (PipelineToken::from(id), type_id))
                    .collect()
            };
            // `[CLS] $A [SEP]`
            let single = Template {
                prefix: run(&[cls], 0),
                suffix: run(&[sep], 0),
                ..Template::default()
            };
            let pair = if kind == "BertProcessing" {
                // `[CLS] $A [SEP] $B@1 [SEP]@1`
                Template {
                    prefix: run(&[cls], 0),
                    infix: run(&[sep], 0),
                    suffix: run(&[sep], 1),
                    a_type_id: 0,
                    b_type_id: Some(1),
                }
            } else {
                // `<s> $A </s></s> $B </s>`, all at type id 0
                Template {
                    prefix: run(&[cls], 0),
                    infix: run(&[sep, sep], 0),
                    suffix: run(&[sep], 0),
                    a_type_id: 0,
                    b_type_id: Some(0),
                }
            };
            Ok(PipelinePostProcessor { single, pair })
        }
        other => Err(unsupported(&format!("the `{other}` post-processor"))),
    }
}

/// A `["<token>", id]` pair, which is how `BertProcessing` and `RobertaProcessing` spell `cls` and
/// `sep`. Only the id reaches the pipeline; the string is the human-readable half.
fn read_special_id(cfg: &Json<'_>, key: &str) -> Result<u32> {
    cfg.field(key)
        .and_then(Json::as_array)
        .filter(|pair| pair.len() == 2)
        .and_then(|pair| pair[1].as_u32())
        .ok_or_else(|| format!("`{key}` is not a [token, id] pair").into())
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
                    let id = id
                        .as_u32()
                        .ok_or_else(|| -> tk_encode::Error { "a template piece has a bad id".into() })?;
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
            return Err(format!("not supported: `{key}` template references the wrong sequences").into());
        }
        (t.prefix, t.infix, t.suffix) = (prefix.into(), infix.into(), suffix.into());
        Ok(t)
    };

    Ok(PipelinePostProcessor {
        single: template_for("single")?,
        pair: template_for("pair")?,
    })
}
