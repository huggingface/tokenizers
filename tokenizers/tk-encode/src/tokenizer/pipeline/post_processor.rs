//! The post-processor half of the pipeline: the [`Template`] representation, the helper that
//! composes templates, and [`PipelinePostProcessor`] itself.

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

/// Which member of an input pair a sequence refers to.
#[derive(Clone, Copy, Debug)]
pub enum Seq {
    A,
    B,
}

/// A post-processing template, in the only shape there is: `prefix? $A infix? ($B suffix?)?`.
///
/// Nothing else needs supporting. A template references `$A` exactly once, references `$B` only in
/// a pair template and only after `$A`, and every other piece is a special token -- so the three
/// runs of specials are concatenated once, when the template is read, and the encode path never
/// walks a piece list. It appends `infix`, B and `suffix` to sequence A's own buffer and splices
/// `prefix` in front, which is why there is no fast path left to pre-compute: for a single
/// template that reuse is unconditional.
///
/// Specials are `(id, type_id)` pairs because one piece can carry several ids, and adjacent pieces
/// can disagree on the type id -- XLNet's suffix is `<sep>@0 <cls>@2`.
#[derive(Clone, Debug, Default)]
pub struct Template {
    /// Specials before sequence A.
    pub prefix: Box<[(PipelineToken, u8)]>,
    /// Specials between A and B. Always empty in a single template.
    pub infix: Box<[(PipelineToken, u8)]>,
    /// Specials after the last sequence.
    pub suffix: Box<[(PipelineToken, u8)]>,
    pub a_type_id: u8,
    /// `None` in a single template, which has no second sequence.
    pub b_type_id: Option<u8>,
}

impl Template {
    /// How many special tokens the template adds, for sizing the output buffer.
    pub fn n_special(&self) -> usize {
        self.prefix.len() + self.infix.len() + self.suffix.len()
    }

    /// Whether anything is tagged with a non-zero `type_id`, i.e. whether the encoding needs a
    /// `type_ids` buffer at all. Derived rather than stored so it cannot fall out of sync; a
    /// handful of specials is nothing next to an encode.
    pub fn has_type_ids(&self) -> bool {
        self.a_type_id != 0
            || self.b_type_id.is_some_and(|id| id != 0)
            || [&self.prefix, &self.infix, &self.suffix]
                .iter()
                .any(|run| run.iter().any(|&(_, id)| id != 0))
    }
}

/// A pass-through template does nothing: no special tokens, and sequences in the default
/// arrangement (`$A`, or `$A $B` with the default type ids 0 then 1). Such a member is a no-op in a
/// Sequence and is dropped when composing. Anything else adds tokens or reorders/retags.
fn is_pass_through(template: &Template) -> bool {
    template.n_special() == 0
        && template.a_type_id == 0
        && matches!(template.b_type_id, None | Some(1))
}

// TODO: this looks unused
pub fn compose<'a>(templates: impl Iterator<Item = &'a Template>) -> Result<Template> {
    let templates = templates.collect::<Vec<_>>();
    let mut chosen: Option<&Template> = None;
    for template in &templates {
        if is_pass_through(template) {
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
    Ok((*chosen).clone())
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

#[cfg(test)]
mod tests {
    use super::*;

    fn run(type_ids: &[u8]) -> Box<[(PipelineToken, u8)]> {
        type_ids
            .iter()
            .map(|&id| (PipelineToken::from(1), id))
            .collect()
    }

    #[test]
    fn type_ids_are_needed_only_when_something_is_tagged() {
        // `[CLS] $A [SEP]`: everything at 0, so no `type_ids` buffer.
        let bert_single = Template {
            prefix: run(&[0]),
            suffix: run(&[0]),
            ..Template::default()
        };
        assert!(!bert_single.has_type_ids());
        assert_eq!(bert_single.n_special(), 2);
        assert!(is_pass_through(&Template::default()));
        assert!(!is_pass_through(&bert_single));

        // `$A <sep>@0 <cls>@2`: tagged in the suffix, which is the case a piece-walking fast path
        // used to bail out of.
        assert!(
            Template {
                suffix: run(&[0, 2]),
                ..Template::default()
            }
            .has_type_ids()
        );
        // `[CLS] $A [SEP] $B [SEP]@1`: tagged through B.
        assert!(
            Template {
                b_type_id: Some(1),
                ..Template::default()
            }
            .has_type_ids()
        );
    }
}
