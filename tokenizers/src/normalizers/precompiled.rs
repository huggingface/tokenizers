use crate::tokenizer::{NormalizedString, Normalizer, Result};
pub use spm_precompiled::Precompiled;
use std::cmp::Ordering;
use unicode_segmentation::UnicodeSegmentation;

fn replace(transformations: &mut Vec<(char, isize)>, old_part: &str, new_part: &str) {
    let old_count = old_part.chars().count() as isize;
    let new_count = new_part.chars().count() as isize;
    let diff = new_count - old_count;

    // If we are just replacing characters, all changes should be == 0
    transformations.extend(new_part.chars().map(|c| (c, 0)));

    match diff.cmp(&0) {
        // If we are adding some characters, the last DIFF characters shoud be == 1
        Ordering::Greater => {
            transformations
                .iter_mut()
                .rev()
                .take(diff as usize)
                .for_each(|(_, cs)| *cs = 1);
        }
        // If we are removing some characters, the last one should include the diff
        Ordering::Less => {
            if let Some((_, cs)) = transformations.last_mut() {
                *cs += diff;
            }
        }
        _ => {}
    }
}

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
                    replace(&mut transformations, grapheme, norm);
                    return;
                }
            }
            for (char_index, c) in grapheme.char_indices() {
                let part = &grapheme[char_index..char_index + c.len_utf8()];
                if let Some(norm) = self.transform(part) {
                    modified = true;
                    replace(&mut transformations, part, norm);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expansion_followed_by_removal() {
        // Simulate transformations from "™\x1eg" to "TMg"
        let mut transformations = vec![];

        let mut n = NormalizedString::from("™\x1eg");
        replace(&mut transformations, "™", "TM");
        replace(&mut transformations, "\x1e", "");
        transformations.push(('g', 0));

        n.transform(transformations.into_iter(), 0);

        assert_eq!(n.get(), "TMg");
    }
}
