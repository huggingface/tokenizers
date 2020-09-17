use crate::tokenizer::{NormalizedString, Normalizer, Result};
pub use spm_precompiled::Precompiled;
use unicode_segmentation::UnicodeSegmentation;

impl Normalizer for Precompiled {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        let mut transformations = Vec::with_capacity(normalized.get().len());
        // Future reader. From @Narsil.
        // Yes, this is weird,
        // Yes, this seems broken
        // No, I don't know why Google did this.
        // If you question this code, check this normalizer against
        // XNLI database (all languages) with Unigram model against
        // Mbart, XLMRoberta *AND* Marian. If you don't get 100% or
        // break a single test.
        // You don't pass.
        let mut modified = false;
        normalized.get().graphemes(true).for_each(|grapheme| {
            if grapheme.len() < 6 {
                if let Some(norm) = self.transform(grapheme) {
                    modified = true;
                    let old_count = grapheme.chars().count() as isize;
                    let new_count = norm.chars().count() as isize;
                    for (i, c) in norm.chars().enumerate() {
                        let n = if i == 0 {
                            new_count - old_count
                        } else {
                            i as isize
                        };
                        transformations.push((c, n));
                    }
                    return;
                }
            }
            for (char_index, c) in grapheme.char_indices() {
                let part = &grapheme[char_index..char_index + c.len_utf8()];
                if let Some(norm) = self.transform(part) {
                    modified = true;
                    let old_count = part.chars().count() as isize;
                    let new_count = norm.chars().count() as isize;
                    for (i, c) in norm.chars().enumerate() {
                        let n = if i == 0 {
                            new_count - old_count
                        } else {
                            i as isize
                        };
                        transformations.push((c, n));
                    }
                } else {
                    transformations.push((c, 0));
                }
            }
        });
        if modified {
            normalized.transform(transformations.into_iter(), 0);
        }
        Ok(())
    }
}
