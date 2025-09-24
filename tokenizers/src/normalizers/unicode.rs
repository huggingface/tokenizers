use crate::tokenizer::{NormalizedString, Normalizer, Result};
use crate::utils::macro_rules_attribute;

#[derive(Default, Copy, Clone, Debug)]
#[macro_rules_attribute(impl_serde_type!)]
pub struct NFD;
impl Normalizer for NFD {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        normalized.nfd();
        Ok(())
    }
}

#[derive(Default, Copy, Clone, Debug)]
#[macro_rules_attribute(impl_serde_type!)]
pub struct NFKD;
impl Normalizer for NFKD {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        normalized.nfkd();
        Ok(())
    }
}

#[derive(Default, Copy, Clone, Debug)]
#[macro_rules_attribute(impl_serde_type!)]
pub struct NFC;
impl Normalizer for NFC {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        normalized.nfc();
        Ok(())
    }
}

#[derive(Default, Copy, Clone, Debug)]
#[macro_rules_attribute(impl_serde_type!)]
pub struct NFKC;
impl Normalizer for NFKC {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        normalized.nfkc();
        Ok(())
    }
}

fn do_nmt(normalized: &mut NormalizedString) {
    // Ascii Control characters
    normalized
        .filter(|c| {
            !matches!(
                c as u32,
                0x0001..=0x0008 |
                0x000B |
                0x000E..=0x001F |
                0x007F |
                0x008F |
                0x009F
            )
        })
        // Other code points considered as whitespace.
        .map(|c| match c as u32 {
            0x0009 => ' ',
            0x000A => ' ',
            0x000C => ' ',
            0x000D => ' ',
            0x1680 => ' ',
            0x200B..=0x200F => ' ',
            0x2028 => ' ',
            0x2029 => ' ',
            0x2581 => ' ',
            0xFEFF => ' ',
            0xFFFD => ' ',
            _ => c,
        });
}

#[derive(Default, Copy, Clone, Debug)]
#[macro_rules_attribute(impl_serde_type!)]
pub struct Nmt;
impl Normalizer for Nmt {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        do_nmt(normalized);
        Ok(())
    }
}

use unicode_general_category::{get_general_category, GeneralCategory};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnicodeFilter {
    filter_unassigned: bool,
    filter_private_use: bool,
}

impl Default for UnicodeFilter {
    fn default() -> Self {
        Self {
            filter_unassigned: true,
            filter_private_use: true,
        }
    }
}

impl UnicodeFilter {
    /// Filters unicode characters based on their general category.
    /// Args:
    ///    filter_unassigned: Whether to filter out unassigned unicode characters
    ///    filter_private_use: Whether to filter out private use unicode characters
    pub fn new(filter_unassigned: bool, filter_private_use: bool) -> Self {
        Self {
            filter_unassigned,
            filter_private_use,
        }
    }
}

impl Normalizer for UnicodeFilter {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        normalized.filter(|c| {
            let category = get_general_category(c);
            !(self.filter_unassigned && category == GeneralCategory::Unassigned ||
              self.filter_private_use && category == GeneralCategory::PrivateUse)
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unicode_filter() {
        // Test with default settings (filter all categories)
        let original = "A\u{20AC}\u{10FFFF}\u{E000}".to_string(); // Regular + Euro + Unassigned + Private Use
        let normalized = "A\u{20AC}".to_string(); // Keep only valid chars
        let mut n = NormalizedString::from(original.clone());
        UnicodeFilter::default().normalize(&mut n).unwrap();
        assert_eq!(n.get(), normalized);

        // Test with only filtering unassigned
        let mut n = NormalizedString::from(original);
        UnicodeFilter::new(true, false).normalize(&mut n).unwrap();
        assert_eq!(n.get(), format!("A\u{20AC}\u{E000}")); // Keep private use, filter unassigned
    }

    #[test]
    fn test_nfkc() {
        let original = "\u{fb01}".to_string();
        let normalized = "fi".to_string();
        let mut n = NormalizedString::from(original.clone());
        NFKC.normalize(&mut n).unwrap();

        assert_eq!(
            n,
            NormalizedString::new(original, normalized, vec![(0, 3), (0, 3)], 0)
        );

        assert_eq!(n.alignments_original(), vec![(0, 2), (0, 2), (0, 2)]);
    }
}
