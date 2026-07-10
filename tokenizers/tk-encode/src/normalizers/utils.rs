use std::borrow::Cow;

use serde::{Deserialize, Serialize};

use crate::normalizers::NormalizerWrapper;
use crate::pipeline;
use crate::tokenizer::{NormalizedString, Normalizer, Result};
use crate::utils::macro_rules_attribute;

#[derive(Clone, Deserialize, Debug, Serialize)]
#[serde(tag = "type")]
/// Allows concatenating multiple other Normalizer as a Sequence.
/// All the normalizers run in sequence in the given order against the same NormalizedString.
pub struct Sequence {
    normalizers: Vec<NormalizerWrapper>,
}

impl Sequence {
    pub fn new(normalizers: Vec<NormalizerWrapper>) -> Self {
        Self { normalizers }
    }
}

impl AsRef<[NormalizerWrapper]> for Sequence {
    fn as_ref(&self) -> &[NormalizerWrapper] {
        &self.normalizers
    }
}

impl AsMut<[NormalizerWrapper]> for Sequence {
    fn as_mut(&mut self) -> &mut [NormalizerWrapper] {
        &mut self.normalizers
    }
}

impl IntoIterator for Sequence {
    type Item = NormalizerWrapper;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.normalizers.into_iter()
    }
}

impl Normalizer for Sequence {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        for normalizer in &self.normalizers {
            normalizer.normalize(normalized)?;
        }
        Ok(())
    }
}

impl pipeline::Normalizer for Sequence {
    fn normalize<'a>(&self, input: &'a str) -> Result<Cow<'a, str>> {
        // Fused metaspace path: `Sequence[Prepend(p), Replace(String s -> c)]` (Llama/Yi/Gemma/Phi-3).
        // Prepend then Replace is two passes + two allocations; fuse into one. Restricted to a
        // single-char `s` so a match can never span the p|input boundary — then replacing within `p`
        // and within `input` separately is byte-exact with replacing `p + input`. (Prepend is a no-op
        // on empty input, so the general path handles that case.)
        if let [NormalizerWrapper::Prepend(p), NormalizerWrapper::Replace(r)] =
            self.normalizers.as_slice()
        {
            if let crate::normalizers::replace::ReplacePattern::String(s) = r.pattern() {
                if !input.is_empty() && !p.prepend.is_empty() && s.chars().count() == 1 {
                    use crate::normalizers::replace::push_str_replaced;
                    let mut out = String::with_capacity(p.prepend.len() + input.len());
                    push_str_replaced(&mut out, &p.prepend, s, &r.content);
                    push_str_replaced(&mut out, input, s, &r.content);
                    return Ok(Cow::Owned(out));
                }
            }
        }

        let mut cow: Cow<'a, str> = Cow::Borrowed(input);
        for normalizer in &self.normalizers {
            cow = match cow {
                // Still borrowing `input` ('a): chain directly so an all-no-op
                // sequence stays zero-alloc and returns a borrow of `input`.
                Cow::Borrowed(s) => pipeline::Normalizer::normalize(normalizer, s)?,
                // Owned locally: the next step may borrow from it, so materialize
                // its result before the local `String` is dropped.
                Cow::Owned(s) => {
                    let out = match pipeline::Normalizer::normalize(normalizer, &s)? {
                        Cow::Owned(o) => Some(o),
                        Cow::Borrowed(b) if b.as_ptr() == s.as_ptr() && b.len() == s.len() => None,
                        Cow::Borrowed(b) => Some(b.to_owned()),
                    };
                    Cow::Owned(out.unwrap_or(s))
                }
            };
        }
        Ok(cow)
    }
}

/// Lowercases the input
#[derive(Copy, Clone, Debug)]
#[macro_rules_attribute(impl_serde_type!)]
pub struct Lowercase;
impl Normalizer for Lowercase {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        normalized.lowercase();
        Ok(())
    }
}

impl pipeline::Normalizer for Lowercase {
    fn normalize<'a>(&self, input: &'a str) -> Result<Cow<'a, str>> {
        // Only LOWER-flagged runs are lowercased; a caseless script is fully inert → borrowed, with no
        // per-char `to_lowercase` probe (the old `.all(lowercases_to_self)` scanned every char).
        use atomsplit::norm_classify::bit;
        Ok(crate::normalizers::tagged::tag_driven(
            input,
            bit::LOWER,
            |run, out| out.extend(run.chars().flat_map(char::to_lowercase)),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalizers::{Strip, StripAccents, NFD};

    #[test]
    fn pipeline_lowercase_matches_legacy() {
        let n = Lowercase;
        for input in &["HELLO", "Hello World", "abc", "", "ÀÉ", "ΟΔΟΣ"] {
            let mut ns = NormalizedString::from(*input);
            Normalizer::normalize(&n, &mut ns).unwrap(); // legacy oracle
            assert_eq!(
                ns.get(),
                &*pipeline::Normalizer::normalize(&n, input).unwrap()
            );
        }
    }

    #[test]
    fn pipeline_sequence_matches_legacy() {
        let n = Sequence::new(vec![NFD.into(), StripAccents.into(), Lowercase.into()]);
        for input in &["Café", "HÉLLO", "abc", "", "ΟΔΟΣ"] {
            let mut ns = NormalizedString::from(*input);
            Normalizer::normalize(&n, &mut ns).unwrap(); // legacy oracle
            assert_eq!(
                ns.get(),
                &*pipeline::Normalizer::normalize(&n, input).unwrap()
            );
        }
    }

    #[test]
    fn pipeline_sequence_metaspace_fusion_matches_legacy() {
        use crate::normalizers::{Prepend, Replace};
        // The fused metaspace path + a prepend that itself contains the pattern + a multi-char pattern
        // (which must fall to the general path) — all must match sequential Prepend-then-Replace.
        let cases: &[(Sequence, &[&str])] = &[
            (
                Sequence::new(vec![
                    Prepend::new("▁".to_string()).into(),
                    Replace::new(" ", "▁").unwrap().into(),
                ]),
                &[
                    "Hello world",
                    " leading",
                    "trailing ",
                    "  多  spaces  ",
                    "nospace",
                    "",
                    "a b c",
                    "▁already",
                ],
            ),
            // prepend contains the pattern char → its internal match must also be replaced
            (
                Sequence::new(vec![
                    Prepend::new(" x".to_string()).into(),
                    Replace::new(" ", "_").unwrap().into(),
                ]),
                &["hi there", "", "no"],
            ),
            // multi-char pattern → not fused (general Cow-threaded path), must still match
            (
                Sequence::new(vec![
                    Prepend::new("<s>".to_string()).into(),
                    Replace::new("ab", "X").unwrap().into(),
                ]),
                &["abab cab", "no match", ""],
            ),
        ];
        for (seq, inputs) in cases {
            for input in *inputs {
                let mut ns = NormalizedString::from(*input);
                Normalizer::normalize(seq, &mut ns).unwrap(); // legacy oracle
                assert_eq!(
                    ns.get(),
                    &*pipeline::Normalizer::normalize(seq, input).unwrap(),
                    "sequence pipeline diverges on {input:?}",
                );
            }
        }
    }

    #[test]
    fn pipeline_sequence_strip_after_owned_matches_legacy() {
        // Strip returns a sub-borrow of the previous step's owned output —
        // the one case where a Borrowed result must not be mistaken for a no-op
        let n = Sequence::new(vec![Lowercase.into(), Strip::new(true, true).into()]);
        for input in &[
            "  HELLO  ",
            "\tMiXeD Case\n",
            "NOPAD",
            "  hello  ",
            "",
            "   ",
        ] {
            let mut ns = NormalizedString::from(*input);
            Normalizer::normalize(&n, &mut ns).unwrap(); // legacy oracle
            assert_eq!(
                ns.get(),
                &*pipeline::Normalizer::normalize(&n, input).unwrap(),
                "input={input:?}"
            );
        }
    }
}
