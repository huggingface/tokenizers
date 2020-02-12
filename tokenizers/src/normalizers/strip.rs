use crate::tokenizer::{NormalizedString, Normalizer, Result};

pub struct Strip {
    strip_left: bool,
    strip_right: bool,
}

impl Strip {
    pub fn new(strip_left: bool, strip_right: bool) -> Self {
        Strip {
            strip_left,
            strip_right,
        }
    }

    fn strip_left(&self, normalized: &mut NormalizedString) -> Result<()> {
        let mut removed: usize = 0;
        let mut still_looking: bool = true;
        let filtered = normalized
            .get()
            .chars()
            // We need to collect here to be able to reverse the iterator because Char is not ended
            .collect::<Vec<_>>()
            .into_iter()
            .map(|c: char| {
                if still_looking {
                    if c.is_whitespace() {
                        removed += 1;
                        None
                    } else {
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
        let mut filtered = normalized
            .get()
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
                    } else {
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

impl Normalizer for Strip {
    /// Strip the normalized string inplace
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        if self.strip_left {
            self.strip_left(normalized).unwrap();
        }

        if self.strip_right {
            self.strip_right(normalized).unwrap();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_left() {
        let s = &mut NormalizedString::from("  This is an example ");
        let normalizer = Strip::new(true, false);
        if normalizer.normalize(s).is_ok() {
            assert_eq!(s.get(), "This is an example ")
        }
    }

    #[test]
    fn strip_right() {
        let s = &mut NormalizedString::from("  This is an example ");
        let normalizer = Strip::new(false, true);
        if normalizer.normalize(s).is_ok() {
            assert_eq!(s.get(), "  This is an example")
        }
    }

    #[test]
    fn strip_full() {
        let s = &mut NormalizedString::from("  This is an example ");
        let normalizer = Strip::new(true, true);
        if normalizer.normalize(s).is_ok() {
            assert_eq!(s.get(), "This is an example")
        }
    }
}
