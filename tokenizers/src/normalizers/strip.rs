use crate::tokenizer::{Normalizer, NormalizedString, Result};

pub struct StripNormalizer {}

impl StripNormalizer{
    fn strip_left(&self, normalized: &mut NormalizedString) -> Result<()> {
        let mut removed: usize = 0;
        let mut still_looking: bool = true;
        let mut filtered = normalized.get()
            .chars()
            // We need to collect here to be able to reverse the iterator because Char is not ended
            .collect::<Vec<_>>()
            .into_iter()
            .map(|c: char| {
                if still_looking {
                    if c.is_whitespace() {
                        removed += 1;
                        None
                    }else {
                        still_looking = false;
                        Some((c, -(removed as isize)))
                    }
                } else {
                    Some((c, 0))
                }
            })
            .collect::<Vec<_>>();

        normalized.transform(
            filtered.iter().filter(|o| o.is_some()).map(|o| o.unwrap()),
            0,
        );
        Ok(())
    }

    fn strip_right(&self, normalized: &mut NormalizedString) -> Result<()> {
        let mut removed: usize = 0;
        let mut still_looking: bool = true;
        let mut filtered = normalized.get()
            .chars()
            // We need to collect here to be able to reverse the iterator because Char is not ended
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .map(|c: char| {
                if still_looking {
                    if c.is_whitespace() {
                        removed += 1;
                        None
                    }else {
                        still_looking = false;
                        Some((c, -(removed as isize)))
                    }
                } else {
                    Some((c, 0))
                }
            })
            .collect::<Vec<_>>();

        filtered.reverse();
        normalized.transform(
            filtered.iter().filter(|o| o.is_some()).map(|o| o.unwrap()),
            0,
        );
        Ok(())
    }
}

impl Normalizer for StripNormalizer {

    /// Strip the normalized string inplace
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        self.strip_left(normalized);
        self.strip_right(normalized);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_complete() {
        let mut s = &mut NormalizedString::from("  This is an example ");
        let normalizer = StripNormalizer{};
        match normalizer.normalize(s){
            Ok(_) => {
                assert_eq!(s.get(), "This is an example")
            },
            _ => {}
        }
    }
}