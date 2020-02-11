use crate::tokenizer::{Normalizer, NormalizedString, Result};

//pub struct StripNormalizer{
//    strip_right: bool,
//    strip_left: bool
//}
//
//
//impl StripNormalizer{
//    pub fn new(strip_right: bool, strip_left)
//}
pub struct StripNormalizer {}
impl Normalizer for StripNormalizer {

    /// Strip the normalized string inplace
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        let mut original = normalized.get_original().to_string();
        let trimmed = original.trim();
        let t_start = trimmed.as_ptr() as usize - original.as_ptr() as usize;
        let t_length = trimmed.len();

        if t_start != 0 {
            original.drain(..t_start);
        }
        original.truncate(t_length);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_complete() {
        let mut s = &mut NormalizedString::from(" This is an example ");
        let normalizer = StripNormalizer{};
        match normalizer.normalize(s){
            Ok(_) => {
                assert_eq!(s.get_original(), "This is an example")
            },
            _ => {}
        }
    }
}