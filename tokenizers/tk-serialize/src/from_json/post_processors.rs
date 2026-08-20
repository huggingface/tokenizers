//! The `post_processor` object, lowered into the config path's `Template` IR.

use super::unsupported;
use crate::json::Json;
use tk_encode::pipeline::{PipelinePostProcessor, PipelineToken, Seq, Slice, Template, compose};
use tk_encode::tokenizer::Result;

/// The post-processor, lowered into the same `Template` IR the config path produces.
pub(super) fn read_post_processor(cfg: Option<&Json<'_>>) -> Result<PipelinePostProcessor> {
    let Some(cfg) = cfg else {
        return Ok(PipelinePostProcessor::default());
    };
    let kind = cfg
        .type_tag()
        .ok_or_else(|| unsupported("a post-processor with no `type`"))?;
    match kind {
        // ByteLevel only trims offsets, which the pipeline does not track: a pass-through.
        "ByteLevel" => Ok(PipelinePostProcessor::default()),
        "Sequence" => {
            let members = cfg
                .get_some("processors")
                .and_then(Json::as_arr)
                .unwrap_or(&[]);
            let built = members
                .iter()
                .map(|m| read_post_processor(Some(m)))
                .collect::<Result<Vec<_>>>()?;
            // Reuse the config path's composition rules rather than restating them.
            Ok(PipelinePostProcessor::new(
                compose(built.iter().map(|m| m.templates().0))?,
                compose(built.iter().map(|m| m.templates().1))?,
            ))
        }
        "TemplateProcessing" => read_template(cfg),
        // The two frames that predate `TemplateProcessing`. Slice shapes copied from
        // `TryFrom<&PostProcessorWrapper>`; note roberta pairs with a doubled sep and keeps every
        // type id at 0, where bert retags the second sequence.
        "BertProcessing" | "RobertaProcessing" => {
            let cls = read_special_id(cfg, "cls")?;
            let sep = read_special_id(cfg, "sep")?;
            let one = |id: u32, type_id: u8| Slice::Specials {
                tokens: Box::new([PipelineToken::from(id)]),
                type_id,
            };
            let sq = |seq, type_id| Slice::Sequence { seq, type_id };
            let single = Template::new(vec![one(cls, 0), sq(Seq::A, 0), one(sep, 0)]);
            let pair = if kind == "BertProcessing" {
                Template::new(vec![
                    one(cls, 0),
                    sq(Seq::A, 0),
                    one(sep, 0),
                    sq(Seq::B, 1),
                    one(sep, 1),
                ])
            } else {
                Template::new(vec![
                    one(cls, 0),
                    sq(Seq::A, 0),
                    Slice::Specials {
                        tokens: Box::new([PipelineToken::from(sep), PipelineToken::from(sep)]),
                        type_id: 0,
                    },
                    sq(Seq::B, 0),
                    one(sep, 0),
                ])
            };
            Ok(PipelinePostProcessor::new(single, pair))
        }
        other => Err(unsupported(&format!("the `{other}` post-processor"))),
    }
}

/// A `["<token>", id]` pair, which is how `BertProcessing` and `RobertaProcessing` spell `cls` and
/// `sep`. Only the id reaches the pipeline; the string is the human-readable half.
fn read_special_id(cfg: &Json<'_>, key: &str) -> Result<u32> {
    cfg.get_some(key)
        .and_then(Json::as_arr)
        .filter(|pair| pair.len() == 2)
        .and_then(|pair| pair[1].as_u32())
        .ok_or_else(|| format!("`{key}` is not a [token, id] pair").into())
}

fn read_template(cfg: &Json<'_>) -> Result<PipelinePostProcessor> {
    // `special_tokens` maps a placeholder to the ids it expands to.
    let specials = cfg
        .get_some("special_tokens")
        .ok_or_else(|| -> tk_encode::Error {
            "TemplateProcessing has no `special_tokens`".into()
        })?;

    let ids_for =
        |name: &str| -> Result<Vec<u32>> {
            let entry = specials.get(name).ok_or_else(|| -> tk_encode::Error {
                format!("template references unknown special token {name:?}").into()
            })?;
            let ids = entry.get_some("ids").and_then(Json::as_arr).ok_or_else(
                || -> tk_encode::Error { format!("special token {name:?} has no `ids`").into() },
            )?;
            ids.iter()
                .map(|i| {
                    i.as_u32().ok_or_else(|| -> tk_encode::Error {
                        format!("special token {name:?} has a bad id").into()
                    })
                })
                .collect()
        };

    let slices_for = |key: &str| -> Result<Vec<Slice>> {
        let pieces =
            cfg.get_some(key)
                .and_then(Json::as_arr)
                .ok_or_else(|| -> tk_encode::Error {
                    format!("TemplateProcessing has no `{key}`").into()
                })?;
        let mut out = Vec::with_capacity(pieces.len());
        for piece in pieces {
            if let Some(seq) = piece.get_some("Sequence") {
                let id = seq.get_some("id").and_then(Json::as_str).ok_or_else(
                    || -> tk_encode::Error { "template Sequence has no `id`".into() },
                )?;
                let type_id = seq.get_some("type_id").and_then(Json::as_u32).unwrap_or(0) as u8;
                let seq = match id {
                    "A" => Seq::A,
                    "B" => Seq::B,
                    other => return Err(format!("unknown template sequence {other:?}").into()),
                };
                out.push(Slice::Sequence { seq, type_id });
            } else if let Some(tok) = piece.get_some("SpecialToken") {
                let id = tok.get_some("id").and_then(Json::as_str).ok_or_else(
                    || -> tk_encode::Error { "template SpecialToken has no `id`".into() },
                )?;
                let type_id = tok.get_some("type_id").and_then(Json::as_u32).unwrap_or(0) as u8;
                out.push(Slice::Specials {
                    tokens: ids_for(id)?.into_iter().map(PipelineToken::from).collect(),
                    type_id,
                });
            } else {
                return Err("a template piece is neither Sequence nor SpecialToken".into());
            }
        }
        Ok(out)
    };

    Ok(PipelinePostProcessor::new(
        Template::new(slices_for("single")?),
        Template::new(slices_for("pair")?),
    ))
}
