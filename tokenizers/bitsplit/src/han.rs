//! `\p{Han}` (Script=Han) for kimi-k2's leading `[\p{Han}]+` arm.
//!
//! - Han is NOT deepseek's `is_cjk_at` range: that one is Han U+4E00..9FA5 plus kana, this one is
//!   the whole script and excludes kana. They are different predicates on purpose.
//! - Every range here is ≥ U+2E80, so a lead byte below 0xE2 exits before any decoding.

/// Script=Han, sorted and disjoint. Extensions past Ext F are included because onig's `\p{Han}`
/// has them; the parity gate is what actually pins this table to the oracle's Unicode version.
const HAN: &[(u32, u32)] = &[
    (0x2E80, 0x2E99),
    (0x2E9B, 0x2EF3),
    (0x2F00, 0x2FD5),
    (0x3005, 0x3005),
    (0x3007, 0x3007),
    (0x3021, 0x3029),
    (0x3038, 0x303B),
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFA6D),
    (0xFA70, 0xFAD9),
    (0x20000, 0x2A6DF),
    (0x2A700, 0x2B739),
    (0x2B740, 0x2B81D),
    (0x2B820, 0x2CEA1),
    (0x2CEB0, 0x2EBE0),
    (0x2EBF0, 0x2EE5D),
    (0x2F800, 0x2FA1D),
    (0x30000, 0x3134A),
    (0x31350, 0x323AF),
];

#[inline]
#[must_use]
pub(crate) fn is_han(cp: u32) -> bool {
    HAN.binary_search_by(|&(lo, hi)| {
        if cp < lo {
            core::cmp::Ordering::Greater
        } else if cp > hi {
            core::cmp::Ordering::Less
        } else {
            core::cmp::Ordering::Equal
        }
    })
    .is_ok()
}

/// Is the char whose lead byte is at `p` in Script=Han? `false` on a truncated tail.
#[inline]
#[must_use]
pub(crate) fn is_han_at(text: &[u8], p: usize) -> bool {
    let b = text[p];
    let n = text.len();
    let cp = if (0xE2..=0xEF).contains(&b) {
        if p + 2 >= n {
            return false;
        }
        ((b as u32 & 0x0F) << 12)
            | ((text[p + 1] as u32 & 0x3F) << 6)
            | (text[p + 2] as u32 & 0x3F)
    } else if (0xF0..=0xF4).contains(&b) {
        if p + 3 >= n {
            return false;
        }
        ((b as u32 & 0x07) << 18)
            | ((text[p + 1] as u32 & 0x3F) << 12)
            | ((text[p + 2] as u32 & 0x3F) << 6)
            | (text[p + 3] as u32 & 0x3F)
    } else {
        return false;
    };
    is_han(cp)
}
