//! The post-processor half of the pipeline: two [`Template`]s and the helper that composes them.

use crate::tokenizer::Result;

use super::{Encoding, PipelineToken};

/// The templates to weave around one sequence and around a pair.
#[derive(Debug)]
pub struct PipelinePostProcessor {
    pub single: Template,
    pub pair: Template,
}

impl Default for PipelinePostProcessor {
    fn default() -> Self {
        Self {
            single: Template::default(),
            pair: Template {
                b_type_id: Some(1),
                ..Template::default()
            },
        }
    }
}

/// Which member of an input pair a sequence refers to.
#[derive(Clone, Copy, Debug)]
pub enum Seq {
    A,
    B,
}

/// A template, in the only shape there is: `prefix? $A infix? ($B suffix?)?`.
///
/// `$A` appears exactly once, `$B` only in a pair template and only after it, and every other
/// piece is a special token -- so the three runs are concatenated when the template is read and
/// the encode path never walks a piece list. Specials are `(id, type_id)` pairs because one piece
/// can carry several ids and adjacent pieces can disagree on the type id (XLNet: `<sep>@0 <cls>@2`).
#[derive(Clone, Debug, Default)]
pub struct Template {
    pub prefix: Box<[(PipelineToken, u8)]>,
    /// Between A and B, so always empty in a single template.
    pub infix: Box<[(PipelineToken, u8)]>,
    /// After the last sequence.
    pub suffix: Box<[(PipelineToken, u8)]>,
    pub a_type_id: u8,
    /// `None` in a single template, which has no second sequence.
    pub b_type_id: Option<u8>,
}

impl Template {
    /// How many special tokens this adds, for sizing the output buffer.
    pub fn n_special(&self) -> usize {
        self.prefix.len() + self.infix.len() + self.suffix.len()
    }

    /// Whether anything carries a non-zero `type_id`, i.e. whether the encoding needs a `type_ids`
    /// buffer at all. Derived, so it cannot fall out of sync; a handful of specials is nothing
    /// next to an encode.
    pub fn has_type_ids(&self) -> bool {
        self.a_type_id != 0
            || self.b_type_id.is_some_and(|id| id != 0)
            || [&self.prefix, &self.infix, &self.suffix]
                .iter()
                .any(|run| run.iter().any(|&(_, id)| id != 0))
    }

    /// Weave the specials around the encoded sequences.
    ///
    /// `SPECIALS` is a const parameter so `add_special_tokens` is a constant in here, not a branch.
    pub(super) fn post_process<const SPECIALS: bool>(
        &self,
        s1: Vec<PipelineToken>,
        s2: Option<Vec<PipelineToken>>,
    ) -> Encoding {
        let (a_len, b_len) = (s1.len(), s2.as_ref().map_or(0, Vec::len));

        let type_ids = self.has_type_ids().then(|| {
            let mut out = Vec::with_capacity(self.n_special() + a_len + b_len);
            if SPECIALS {
                out.extend(self.prefix.iter().map(|&(_, id)| id));
            }
            out.resize(out.len() + a_len, self.a_type_id);
            if SPECIALS {
                out.extend(self.infix.iter().map(|&(_, id)| id));
            }
            if let Some(id) = self.b_type_id {
                out.resize(out.len() + b_len, id);
            }
            if SPECIALS {
                out.extend(self.suffix.iter().map(|&(_, id)| id));
            }
            out
        });

        let ids = match s2 {
            Some(b) => self.weave_pair::<SPECIALS>(s1, b),
            None => {
                // A's buffer already holds the front of the answer, so keep it.
                debug_assert!(self.infix.is_empty(), "[BUG] single template with an infix");
                let mut ids = s1;
                if SPECIALS {
                    ids.reserve(self.n_special());
                    ids.extend(self.suffix.iter().map(|&(id, _)| id));
                    // TODO: hand A's buffer `prefix.len()` free slots up front and this becomes a
                    // `copy_from_slice`, with `SPECIALS == false` returning a start offset.
                    if !self.prefix.is_empty() {
                        ids.splice(0..0, self.prefix.iter().map(|&(id, _)| id));
                    }
                }
                ids
            }
        };
        Encoding::new(ids, type_ids)
    }

    /// One exact allocation. A pair cannot reuse A's buffer -- that would do all these same copies
    /// and then memmove A and B aside for the prefix -- and inlining this measurably slows the
    /// single-sequence path, which is the common one.
    #[inline(never)]
    fn weave_pair<const SPECIALS: bool>(
        &self,
        a: Vec<PipelineToken>,
        b: Vec<PipelineToken>,
    ) -> Vec<PipelineToken> {
        let mut ids = Vec::with_capacity(self.n_special() + a.len() + b.len());
        if SPECIALS {
            ids.extend(self.prefix.iter().map(|&(id, _)| id));
        }
        ids.extend(a);
        if SPECIALS {
            ids.extend(self.infix.iter().map(|&(id, _)| id));
        }
        ids.extend(b);
        if SPECIALS {
            ids.extend(self.suffix.iter().map(|&(id, _)| id));
        }
        ids
    }

    /// Adds nothing and retags nothing, so it is a no-op member of a `Sequence`.
    fn is_pass_through(&self) -> bool {
        self.n_special() == 0 && self.a_type_id == 0 && matches!(self.b_type_id, None | Some(1))
    }
}

/// The one member of a `Sequence` post-processor that actually does something. Falls back to the
/// first, so an all-pass-through `Sequence` keeps that member's sequence arrangement.
pub fn compose<'a>(templates: impl Iterator<Item = &'a Template>) -> Result<Template> {
    let (mut first, mut chosen) = (None, None);
    for template in templates {
        first = first.or(Some(template));
        if template.is_pass_through() {
            continue;
        }
        if chosen.replace(template).is_some() {
            return Err(
                "post processor Sequence with multiple sequence referencing members is not supported".into(),
            );
        }
    }
    chosen
        .or(first)
        .cloned()
        .ok_or_else(|| "empty Sequence post processor is not supported".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(type_ids: &[u8]) -> Box<[(PipelineToken, u8)]> {
        type_ids.iter().map(|&id| (1.into(), id)).collect()
    }

    #[test]
    fn type_ids_are_needed_only_when_something_is_tagged() {
        // `[CLS] $A [SEP]`, all at 0: no `type_ids` buffer, but not a no-op either.
        let bert = Template {
            prefix: run(&[0]),
            suffix: run(&[0]),
            ..Template::default()
        };
        assert!(!bert.has_type_ids() && !bert.is_pass_through());
        assert_eq!(bert.n_special(), 2);
        assert!(Template::default().is_pass_through());
        // `$A <sep>@0 <cls>@2`, tagged in the suffix, and `... $B@1 ...`, tagged through B.
        for tagged in [
            Template {
                suffix: run(&[0, 2]),
                ..Template::default()
            },
            Template {
                b_type_id: Some(1),
                ..Template::default()
            },
        ] {
            assert!(tagged.has_type_ids());
        }
    }
}
