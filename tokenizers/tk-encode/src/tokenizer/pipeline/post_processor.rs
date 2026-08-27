//! The post-processor half of the pipeline: the [`Template`] representation
//! ([`Slice`] / [`Seq`]), the lowering helpers that build and compose templates, and
//! [`PipelinePostProcessor`] itself.

use std::convert::TryFrom;

use crate::pipeline::Encoding;
use crate::processors::template::{Piece, Sequence, Tokens};
use crate::tokenizer::Result;

use super::PipelineToken;

/// A post-processor compiled to a prefix and a suffix (slices of token IDs)
///
/// Example:
/// ```text
///     PipelinePostProcessor {
///         prefix: vec![100].into_boxed_slice(),
///         suffix: vec![101, 102].into_boxed_slice()
///     };
///
///     [CLS] The quick Brown fox  [SEP]
///     <100>|  <3> <4> <19> <67> | <101> <102>
///   prefix |  sequence encoding | suffix
/// ```
///
#[derive(Debug)]
pub struct PipelinePostProcessor {
    pub(super) single: Template,
    pub(super) pair: Template,
}

impl PipelinePostProcessor {
    /// Assemble one from an already-built single and pair template.
    ///
    /// Exists so both readers stop at the same door: the slim reader builds the two templates from
    /// JSON, `tk-convert` builds them from a `PostProcessorWrapper`, and neither needs the
    /// fields.
    pub fn new(single: Template, pair: Template) -> Self {
        Self { single, pair }
    }

    /// The two templates back out again, for composing a `Sequence` post-processor out of the
    /// members it lowered to. Read-only: `compose` picks one of them, it does not edit them.
    pub fn templates(&self) -> (&Template, &Template) {
        (&self.single, &self.pair)
    }
}

#[derive(Clone, Debug)]
pub enum Slice {
    Specials {
        tokens: Box<[PipelineToken]>,
        type_id: u8,
    },
    Sequence {
        seq: Seq,
        type_id: u8,
    },
}

#[derive(Clone, Copy, Debug)]
pub enum Seq {
    A,
    B,
}

#[derive(Debug)]
pub struct Template {
    pub(super) slices: Box<[Slice]>,
    pub(super) n_special: usize,
    pub(super) has_type_ids: bool,
}

impl Template {
    pub fn new(slices: Vec<Slice>) -> Self {
        let n_special = slices
            .iter()
            .map(|s| {
                if let Slice::Specials { tokens, .. } = s {
                    tokens.len()
                } else {
                    0
                }
            })
            .sum();
        let has_type_ids = slices.iter().any(|s| match s {
            Slice::Specials { type_id, .. } | Slice::Sequence { type_id, .. } => *type_id != 0,
        });
        Self {
            slices: slices.into_boxed_slice(),
            n_special,
            has_type_ids,
        }
    }

    pub fn slices(&self) -> &[Slice] {
        &self.slices
    }

    pub fn single_sequence_is_noop(&self, add_special_tokens: bool) -> bool {
        if self.has_type_ids {
            return false;
        }
        let mut count_sequences = 0usize;
        let only_sequence_a = self.slices.iter().all(|slice| match slice {
            Slice::Specials { .. } => !add_special_tokens,
            Slice::Sequence { seq: Seq::A, .. } => {
                count_sequences += 1;
                true
            }
            Slice::Sequence { seq: Seq::B, .. } => false,
        });
        only_sequence_a && count_sequences == 1
    }

    pub fn apply(
        &self,
        sequence_a: &[PipelineToken],
        maybe_sequence_b: Option<&[PipelineToken]>,
        add_special_tokens: bool,
    ) -> Result<Encoding> {
        let seq_len = sequence_a.len() + maybe_sequence_b.map_or(0, |tokens| tokens.len());
        let total_len = self.n_special + seq_len;

        let mut ids = Vec::with_capacity(total_len);
        let mut type_ids = self.has_type_ids.then(|| Vec::with_capacity(total_len));

        for slice in &self.slices {
            match slice {
                Slice::Specials { tokens, type_id } => {
                    if !add_special_tokens {
                        continue;
                    }
                    ids.extend_from_slice(tokens);
                    if let Some(type_ids) = type_ids.as_mut() {
                        type_ids.resize(type_ids.len() + tokens.len(), *type_id);
                    }
                }
                Slice::Sequence { seq, type_id } => {
                    let tokens = match seq {
                        Seq::A => sequence_a,
                        Seq::B => maybe_sequence_b.ok_or(
                            "[BUG] only a pair template references sequence B, and a pair always provides it",
                        )?,
                    };
                    if let Some(type_ids) = type_ids.as_mut() {
                        type_ids.resize(type_ids.len() + tokens.len(), *type_id);
                    }
                    ids.extend_from_slice(tokens);
                }
            }
        }

        Ok(Encoding::new(ids, type_ids))
    }
}

// TODO: I don't think this is as optimized as can be yet, but we'll do that post v1.
pub fn build_slices(pieces: &[Piece], specials: &Tokens, is_pair: bool) -> Result<Vec<Slice>> {
    let (mut seen_a, mut seen_b) = (false, false);
    let mut slices = Vec::new();
    for piece in pieces {
        match piece {
            Piece::Sequence {
                id: Sequence::A,
                type_id,
            } => {
                if seen_a {
                    return Err(
                        "not supported: template references sequence A more than once".into(),
                    );
                }
                seen_a = true;
                slices.push(Slice::Sequence {
                    seq: Seq::A,
                    type_id: u8::try_from(*type_id)
                        .map_err(|_| "not supported: type_id out of range")?,
                });
            }
            Piece::Sequence {
                id: Sequence::B,
                type_id,
            } => {
                if seen_b {
                    return Err(
                        "not supported: template references sequence B more than once".into(),
                    );
                }
                seen_b = true;
                slices.push(Slice::Sequence {
                    seq: Seq::B,
                    type_id: u8::try_from(*type_id)
                        .map_err(|_| "not supported: type_id out of range")?,
                });
            }
            Piece::SpecialToken {
                id: token_string,
                type_id,
            } => {
                let special = specials.0.get(token_string).ok_or_else(|| {
                    format!("not supported: unknown special token: `{token_string}`")
                })?;
                slices.push(Slice::Specials {
                    tokens: special
                        .ids()
                        .iter()
                        .map(|&id| PipelineToken::from(id))
                        .collect(),
                    type_id: u8::try_from(*type_id)
                        .map_err(|_| "not supported: type_id out of range")?,
                });
            }
        }
    }
    if !seen_a {
        return Err("not supported: template does not reference sequence A".into());
    }
    if is_pair && !seen_b {
        return Err("not supported: pair template does not reference sequence B".into());
    }
    if !is_pair && seen_b {
        return Err(
            "not supported: single template references sequence B (it should only refer to A)"
                .into(),
        );
    }
    Ok(slices)
}

/// A pass-through template does nothing: no special tokens, and sequences in the default
/// arrangement (`$A`, or `$A $B` with the default type ids 0 then 1). Such a member is a no-op in a
/// Sequence and is dropped when composing. Anything else adds tokens or reorders/retags.
fn is_pass_through(slices: &[Slice]) -> bool {
    matches!(
        slices,
        [Slice::Sequence {
            seq: Seq::A,
            type_id: 0
        }] | [
            Slice::Sequence {
                seq: Seq::A,
                type_id: 0
            },
            Slice::Sequence {
                seq: Seq::B,
                type_id: 1
            }
        ]
    )
}

// TODO: this looks unused
pub fn compose<'a>(templates: impl Iterator<Item = &'a Template>) -> Result<Template> {
    let templates = templates.collect::<Vec<_>>();
    let mut chosen: Option<&Template> = None;
    for template in &templates {
        if is_pass_through(&template.slices) {
            continue;
        }
        if chosen.replace(template).is_some() {
            return Err(
                "post processor Sequence with multiple sequence referencing members is not supported".into(),
            );
        }
    }
    let chosen = chosen
        .or_else(|| templates.first().copied())
        .ok_or("empty Sequence post processor is not supported")?;
    Ok(Template::new(chosen.slices.to_vec()))
}

impl Default for PipelinePostProcessor {
    fn default() -> Self {
        Self {
            single: Template::new(vec![Slice::Sequence {
                seq: Seq::A,
                type_id: 0,
            }]),
            pair: Template::new(vec![
                Slice::Sequence {
                    seq: Seq::A,
                    type_id: 0,
                },
                Slice::Sequence {
                    seq: Seq::B,
                    type_id: 1,
                },
            ]),
        }
    }
}
