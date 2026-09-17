//! The post-processor half of the pipeline: the two [`Template`]s an encode adds its tokens from.

use super::PipelineToken;

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
}
