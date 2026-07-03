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
    fn normalize<'a>(&self, input: &'a str, scratch: &'a mut String) -> &'a str {
        let Some((first_normalizer, rest)) = self.normalizers.split_first() else {
            return input;
        };

        // A normalizer can't read and write the same buffer, so chaining needs two.
        // `scratch` is one; `spare` is the alternate.
        let mut spare = String::new();
        normalize_into(first_normalizer, input, scratch);
        for normalizer in rest {
            normalize_into(normalizer, scratch, &mut spare);
            std::mem::swap(scratch, &mut spare);
        }
        scratch
    }
}

/// Run `n` on `src`, leaving its full output in `dst` (cleared first). A
/// normalizer may hand back a borrow of `src` rather than writing (input left
/// unchanged, or a trimmed sub-slice); we copy that in so the result always
/// lives in `dst`.
fn normalize_into(n: &NormalizerWrapper, src: &str, dst: &mut String) {
    dst.clear();
    // The result's lifetime is tied to the `&mut dst` reborrow, so grab its
    // address/len and drop the borrow before touching `dst` again.
    let (out_ptr, out_len) = {
        let out = pipeline::Normalizer::normalize(n, src, dst);
        (out.as_ptr() as usize, out.len())
    };
    if out_ptr == dst.as_ptr() as usize {
        return; // `n` wrote straight into `dst`
    }
    // Otherwise `out` borrows `src`; copy that slice in.
    let start = out_ptr - src.as_ptr() as usize;
    dst.push_str(&src[start..start + out_len]);
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
    fn normalize<'a>(&self, input: &'a str, scratch: &'a mut String) -> &'a str {
        scratch.clear();
        scratch.extend(input.chars().flat_map(char::to_lowercase));
        scratch
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalizers::{StripAccents, NFD};

    #[test]
    fn pipeline_lowercase_matches_legacy() {
        let n = Lowercase;
        for input in &["HELLO", "Hello World", "abc", "", "ÀÉ"] {
            let mut ns = NormalizedString::from(*input);
            Normalizer::normalize(&n, &mut ns).unwrap(); // legacy oracle
            let mut scratch = String::new();
            assert_eq!(
                ns.get(),
                &*pipeline::Normalizer::normalize(&n, input, &mut scratch)
            );
        }
    }

    #[test]
    fn pipeline_sequence_matches_legacy() {
        let n = Sequence::new(vec![NFD.into(), StripAccents.into(), Lowercase.into()]);
        for input in &["Café", "HÉLLO", "abc", ""] {
            let mut ns = NormalizedString::from(*input);
            Normalizer::normalize(&n, &mut ns).unwrap(); // legacy oracle
            let mut scratch = String::new();
            assert_eq!(
                ns.get(),
                &*pipeline::Normalizer::normalize(&n, input, &mut scratch)
            );
        }
    }
}
