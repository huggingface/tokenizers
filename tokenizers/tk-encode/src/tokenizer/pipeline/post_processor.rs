//! The post-processor half of the pipeline: the two [`Template`]s an encode adds its tokens from.

use super::{Encoding, PipelineToken};

#[inline]
fn push_ids(run: &[(PipelineToken, u8)], out: &mut Vec<PipelineToken>) {
    out.extend(run.iter().map(|&(id, _)| id));
}

#[inline]
fn push_type_ids(run: &[(PipelineToken, u8)], out: &mut Vec<u8>) {
    out.extend(run.iter().map(|&(_, id)| id));
}

/// The templates to wrap around one sequence and around a pair.
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

/// A template, in the only shape there is: `prefix? $A infix? ($B suffix?)?`.
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

    /// whether the encoding needs a `type_ids` buffer at all
    pub fn has_type_ids(&self) -> bool {
        self.a_type_id != 0
            || self.b_type_id.is_some_and(|id| id != 0)
            || [&self.prefix, &self.infix, &self.suffix]
                .iter()
                .any(|run| run.iter().any(|&(_, id)| id != 0))
    }

    /// Type ids for `a_len` ids of A and `b_len` of B; `None` when all 0.
    fn type_ids<const SPECIALS: bool>(&self, a_len: usize, b_len: usize) -> Option<Vec<u8>> {
        self.has_type_ids().then(|| {
            let mut out = Vec::with_capacity(self.n_special() + a_len + b_len);
            if SPECIALS {
                push_type_ids(&self.prefix, &mut out);
            }
            out.resize(out.len() + a_len, self.a_type_id);
            if SPECIALS {
                push_type_ids(&self.infix, &mut out);
            }
            if let Some(id) = self.b_type_id {
                out.resize(out.len() + b_len, id);
            }
            if SPECIALS {
                push_type_ids(&self.suffix, &mut out);
            }
            out
        })
    }

    pub fn post_process<const SPECIALS: bool>(
        &self,
        s1: Vec<PipelineToken>,
        s2: Option<Vec<PipelineToken>>,
    ) -> Encoding {
        let Some(b) = s2 else {
            // A's buffer already holds the front of the answer, so keep it.
            debug_assert!(self.infix.is_empty(), "[BUG] single template with an infix");
            let type_ids = self.type_ids::<SPECIALS>(s1.len(), 0);
            let mut ids = s1;
            if SPECIALS {
                ids.reserve(self.n_special());
                push_ids(&self.suffix, &mut ids);
                if !self.prefix.is_empty() {
                    ids.splice(0..0, self.prefix.iter().map(|&(id, _)| id));
                }
            }
            return Encoding::new(ids, type_ids, None);
        };
        self.post_process_parts::<SPECIALS>(
            std::slice::from_ref(&s1),
            Some(std::slice::from_ref(&b)),
        )
    }

    /// [`Self::post_process`] for a sequence in parts: each part is copied once into an
    /// exactly-sized output (glue + wrap copied up to three times).
    pub fn post_process_parts<const SPECIALS: bool>(
        &self,
        a: &[Vec<PipelineToken>],
        b: Option<&[Vec<PipelineToken>]>,
    ) -> Encoding {
        let len = |parts: &[Vec<PipelineToken>]| parts.iter().map(Vec::len).sum::<usize>();
        let (a_len, b_len) = (len(a), b.map_or(0, len));
        debug_assert!(
            b.is_some() || self.infix.is_empty(),
            "[BUG] single template with an infix"
        );
        let type_ids = self.type_ids::<SPECIALS>(a_len, b_len);
        let specials = if SPECIALS { self.n_special() } else { 0 };
        let mut ids = Vec::with_capacity(specials + a_len + b_len);
        if SPECIALS {
            push_ids(&self.prefix, &mut ids);
        }
        for part in a {
            ids.extend_from_slice(part);
        }
        if let Some(b) = b {
            if SPECIALS {
                push_ids(&self.infix, &mut ids);
            }
            for part in b {
                ids.extend_from_slice(part);
            }
        }
        if SPECIALS {
            push_ids(&self.suffix, &mut ids);
        }
        Encoding::new(ids, type_ids, None)
    }
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
        assert!(!bert.has_type_ids());
        assert_eq!(bert.n_special(), 2);
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

    fn ids(range: std::ops::Range<u32>) -> Vec<PipelineToken> {
        range.map(PipelineToken::from).collect()
    }

    /// Parts wrap exactly like the glued sequence.
    #[test]
    fn parts_wrap_like_the_glued_sequence() {
        let tagged = |ids: &[u32], type_id: u8| -> Box<[(PipelineToken, u8)]> {
            ids.iter().map(|&id| (id.into(), type_id)).collect()
        };
        let single = Template {
            prefix: tagged(&[100], 0),
            suffix: tagged(&[101, 102], 0),
            ..Template::default()
        };
        let pair = Template {
            prefix: tagged(&[100], 0),
            infix: tagged(&[101], 0),
            suffix: tagged(&[102], 1),
            b_type_id: Some(1),
            ..Template::default()
        };
        let a = vec![ids(0..3), ids(3..3), ids(3..7)];
        let b = vec![ids(10..12), ids(12..20)];

        assert_eq!(
            single.post_process_parts::<true>(&a, None),
            single.post_process::<true>(a.concat(), None)
        );
        assert_eq!(
            single.post_process_parts::<false>(&a, None),
            single.post_process::<false>(a.concat(), None)
        );
        assert_eq!(
            pair.post_process_parts::<true>(&a, Some(&b)),
            pair.post_process::<true>(a.concat(), Some(b.concat()))
        );
        assert_eq!(
            pair.post_process_parts::<false>(&a, Some(&b)),
            pair.post_process::<false>(a.concat(), Some(b.concat()))
        );
    }
}
