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
    let slices_for = |key: &str| -> Result<Vec<Slice>> {
        let pieces = cfg.need("TemplateProcessing", key, Json::as_array)?;
        let mut out = Vec::with_capacity(pieces.len());
        for piece in pieces {
            // `type_id` is the sequence a piece belongs to, and `0` for all but the second half of
            // a pair, so it is written only when it is not that.
            let type_id = piece.field("type_id").and_then(Json::as_u32).unwrap_or(0) as u8;
            if let Some(seq) = piece.field("seq").and_then(Json::as_str) {
                out.push(Slice::Sequence {
                    seq: read_seq(seq)?,
                    type_id,
                });
            } else if let Some(ids) = piece.field("ids").and_then(Json::as_array) {
                let ids = ids
                    .iter()
                    .map(|id| {
                        id.as_u32().ok_or_else(|| -> tk_encode::Error {
                            "a template piece has a bad id".into()
                        })
                    })
                    .collect::<Result<Vec<u32>>>()?;
                out.push(Slice::Specials {
                    tokens: ids.into_iter().map(PipelineToken::from).collect(),
                    type_id,
                });
            } else {
                // The `{"SpecialToken": {"id": ...}}` wrapper and its `special_tokens` table are a
                // `1.0` spelling, and a `1.0` file never reaches this reader.
                return Err("a template piece has neither `seq` nor `ids`".into());
            }
        }
        Ok(out)
    };

    Ok(PipelinePostProcessor::new(
        Template::new(slices_for("single")?),
        Template::new(slices_for("pair")?),
    ))
}

fn read_seq(id: &str) -> Result<Seq> {
    match id {
        "A" => Ok(Seq::A),
        "B" => Ok(Seq::B),
        other => Err(format!("unknown template sequence {other:?}").into()),
    }
}
