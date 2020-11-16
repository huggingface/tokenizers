use crate::tokenizer::{NormalizedString, Normalizer, Result};
use serde::{Deserialize, Serialize};
use unicode_normalization_alignments::char::is_combining_mark;

#[derive(Copy, Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type")]
#[non_exhaustive]
pub struct Strip {
    pub strip_left: bool,
    pub strip_right: bool,
}

impl Strip {
    pub fn new(strip_left: bool, strip_right: bool) -> Self {
        Self {
            strip_left,
            strip_right,
        }
    }
}

impl Normalizer for Strip {
    /// Strip the normalized string inplace
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        if self.strip_left && self.strip_right {
            // Fast path
            normalized.strip();
        } else {
            if self.strip_left {
                normalized.lstrip();
            }

            if self.strip_right {
                normalized.rstrip();
            }
        }

        Ok(())
    }
}

// This normalizer removes combining marks from a normalized string
// It's different from unidecode as it does not attempt to modify
// non ascii languages.
#[derive(Copy, Clone, Debug)]
pub struct StripAccents;
impl_serde_unit_struct!(StripAccentsVisitor, StripAccents);

impl Normalizer for StripAccents {
    /// Strip the normalized string inplace
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        normalized.filter(|c| !is_combining_mark(c));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalizer::NormalizedString;
    use crate::normalizers::Lowercase;
    use crate::normalizers::NFKD;
    use unicode_normalization_alignments::UnicodeNormalization;

    #[test]
    fn test_strip_accents() {
        // Unicode combining char
        let original: String = "Me llamó".nfkd().map(|(c, _)| c).collect();
        let normalized = "Me llamo";
        assert_ne!(original, normalized);
        let mut n = NormalizedString::from(original);
        StripAccents.normalize(&mut n).unwrap();
        assert_eq!(&n.get(), &normalized);

        // Ignores regular ascii
        let original = "Me llamo";
        let normalized = "Me llamo";
        assert_eq!(original, normalized);
        let mut n = NormalizedString::from(original);
        StripAccents.normalize(&mut n).unwrap();
        assert_eq!(&n.get(), &normalized);

        // Does not change chinese
        let original: String = "这很简单".nfkd().map(|(c, _)| c).collect();
        let normalized = "这很简单";
        assert_eq!(original, normalized);
        let mut n = NormalizedString::from(original);
        StripAccents.normalize(&mut n).unwrap();
        assert_eq!(&n.get(), &normalized);
    }

    #[test]
    fn test_vietnamese_bug() {
        let original: String = "ậ…".to_string();
        let normalized = "a...".to_string();
        assert_ne!(original, normalized);
        let mut n = NormalizedString::from(original);
        NFKD.normalize(&mut n).unwrap();
        StripAccents.normalize(&mut n).unwrap();
        assert_eq!(&n.get(), &normalized);
        Lowercase.normalize(&mut n).unwrap();
        assert_eq!(&n.get(), &normalized);

        let original: String = "Cụ thể, bạn sẽ tham gia một nhóm các giám đốc điều hành tổ chức, các nhà lãnh đạo doanh nghiệp, các học giả, chuyên gia phát triển và tình nguyện viên riêng biệt trong lĩnh vực phi lợi nhuận…".to_string();
        let normalized = "cu the, ban se tham gia mot nhom cac giam đoc đieu hanh to chuc, cac nha lanh đao doanh nghiep, cac hoc gia, chuyen gia phat trien va tinh nguyen vien rieng biet trong linh vuc phi loi nhuan...".to_string();
        let mut n = NormalizedString::from(original);
        NFKD.normalize(&mut n).unwrap();
        StripAccents.normalize(&mut n).unwrap();
        Lowercase.normalize(&mut n).unwrap();
        assert_eq!(&n.get(), &normalized);
    }

    #[test]
    fn test_thai_bug() {
        let original = "ำน\u{e49}ำ3ลำ".to_string();
        let normalized = "านา3ลา".to_string();
        assert_ne!(original, normalized);
        let mut n = NormalizedString::from(original);
        NFKD.normalize(&mut n).unwrap();
        StripAccents.normalize(&mut n).unwrap();
        Lowercase.normalize(&mut n).unwrap();
        assert_eq!(&n.get(), &normalized);
    }

    #[test]
    fn test_strip_accents_multiple() {
        let original = "e\u{304}\u{304}\u{304}o";
        let normalized = "eo";
        assert_ne!(original, normalized);
        let mut n = NormalizedString::from(original);
        StripAccents.normalize(&mut n).unwrap();
        assert_eq!(&n.get(), &normalized);
        assert_eq!(
            n,
            NormalizedString::new(
                original.to_string(),
                normalized.to_string(),
                vec![(0, 1), (7, 8)],
                0
            )
        );
        assert_eq!(
            n.alignments_original(),
            vec![
                (0, 1),
                (1, 1),
                (1, 1),
                (1, 1),
                (1, 1),
                (1, 1),
                (1, 1),
                (1, 2)
            ]
        );
    }
}
