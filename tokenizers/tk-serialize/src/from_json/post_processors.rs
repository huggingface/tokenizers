//! The `post_processor` object, lowered into the config path's `Template` IR.

use super::unsupported;
use crate::json::Json;
use tk_encode::pipeline::{PipelinePostProcessor, PipelineToken, Seq, Slice, Template, compose};
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
            Ok(PipelinePostProcessor::new(
                compose(built.iter().map(|m| m.templates().0))?,
                compose(built.iter().map(|m| m.templates().1))?,
            ))
        }
        "TemplateProcessing" => read_template(cfg),
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
    cfg.field(key)
        .and_then(Json::as_array)
        .filter(|pair| pair.len() == 2)
        .and_then(|pair| pair[1].as_u32())
        .ok_or_else(|| format!("`{key}` is not a [token, id] pair").into())
}

fn read_template(cfg: &Json<'_>) -> Result<PipelinePostProcessor> {
    let ids_of = |arr: &[Json<'_>], owner: &str| -> Result<Vec<u32>> {
        arr.iter()
            .map(|i| {
                i.as_u32()
                    .ok_or_else(|| -> tk_encode::Error { format!("{owner} has a bad id").into() })
            })
            .collect()
    };

    // A piece carrying its `ids` needs no table; `special_tokens` is only there for the files that
    // name their runs instead, which is every one written before this crate.
    let ids_for = |name: &str| -> Result<Vec<u32>> {
        let entry = cfg
            .field("special_tokens")
            .and_then(|specials| specials.get(name))
            .ok_or_else(|| -> tk_encode::Error {
                format!("template references unknown special token {name:?}").into()
            })?;
        let owner = format!("special token {name:?}");
        ids_of(entry.need(&owner, "ids", Json::as_array)?, &owner)
    };

    let slices_for = |key: &str| -> Result<Vec<Slice>> {
        let pieces = cfg.need("TemplateProcessing", key, Json::as_array)?;
        let mut out = Vec::with_capacity(pieces.len());
        for piece in pieces {
            if let Some(seq) = piece.field("Sequence") {
                let id = seq.need("template Sequence", "id", Json::as_str)?;
                let type_id = seq.field("type_id").and_then(Json::as_u32).unwrap_or(0) as u8;
                let seq = match id {
                    "A" => Seq::A,
                    "B" => Seq::B,
                    other => return Err(format!("unknown template sequence {other:?}").into()),
                };
                out.push(Slice::Sequence { seq, type_id });
            } else if let Some(tok) = piece.field("SpecialToken") {
                let type_id = tok.field("type_id").and_then(Json::as_u32).unwrap_or(0) as u8;
                let ids = match tok.field("ids").and_then(Json::as_array) {
                    Some(arr) => ids_of(arr, "a template SpecialToken")?,
                    None => ids_for(tok.need("template SpecialToken", "id", Json::as_str)?)?,
                };
                out.push(Slice::Specials {
                    tokens: ids.into_iter().map(PipelineToken::from).collect(),
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
