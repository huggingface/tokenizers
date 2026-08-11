use std::borrow::Cow;

use crate::pipeline;
use crate::tokenizer::Result;
use crate::utils::macro_rules_attribute;
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

impl pipeline::Normalizer for Strip {
    fn normalize<'a>(&self, input: &'a str) -> Result<Cow<'a, str>> {
        let s = if self.strip_left {
            input.trim_start()
        } else {
            input
        };
        let s = if self.strip_right { s.trim_end() } else { s };
        Ok(Cow::Borrowed(s))
    }
}

// This normalizer removes combining marks from a normalized string
// It's different from unidecode as it does not attempt to modify
// non ascii languages.
#[derive(Copy, Clone, Debug)]
#[macro_rules_attribute(impl_serde_type!)]
pub struct StripAccents;

impl pipeline::Normalizer for StripAccents {
    fn normalize<'a>(&self, input: &'a str) -> Result<Cow<'a, str>> {
        if input.chars().any(is_combining_mark) {
            Ok(Cow::Owned(
                input.chars().filter(|&c| !is_combining_mark(c)).collect(),
            ))
        } else {
            Ok(Cow::Borrowed(input))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalizers::{Lowercase, NFKD, Sequence, assert_normalizes};

    #[test]
    fn strip_accents_drops_combining_marks() {
        assert_normalizes(
            &StripAccents,
            &[
                // A combining mark only shows up once the text is decomposed, so a
                // composed "café" comes out unchanged: `NFKD` is what feeds this
                // normalizer its marks (see `decomposed_accents_are_stripped`).
                ("café", "café"),
                ("å ç ñ", "å ç ñ"),
                ("e\u{304}\u{304}\u{304}o", "eo"),
                // Han characters carry no marks
                ("这很简单", "这很简单"),
                ("abc", "abc"),
                ("     hello", "     hello"),
                ("", ""),
            ],
        );
    }

    #[test]
    fn decomposed_accents_are_stripped() {
        let n = Sequence::new(vec![NFKD.into(), StripAccents.into(), Lowercase.into()]);
        assert_normalizes(
            &n,
            &[
                ("Me llamó", "me llamo"),
                // Vietnamese: a base letter with two stacked marks, and "…" which NFKD
                // expands to three dots
                ("ậ…", "a..."),
                (
                    "Cụ thể, bạn sẽ tham gia một nhóm các giám đốc điều hành tổ chức, các nhà lãnh đạo doanh nghiệp, các học giả, chuyên gia phát triển và tình nguyện viên riêng biệt trong lĩnh vực phi lợi nhuận…",
                    "cu the, ban se tham gia mot nhom cac giam đoc đieu hanh to chuc, cac nha lanh đao doanh nghiep, cac hoc gia, chuyen gia phat trien va tinh nguyen vien rieng biet trong linh vuc phi loi nhuan...",
                ),
                // Thai: the marks are vowels and tone marks, and dropping them leaves
                // the consonants behind
                ("ำน\u{e49}ำ3ลำ", "านา3ลา"),
            ],
        );
    }

    #[test]
    fn strip_trims_both_sides() {
        assert_normalizes(
            &Strip::new(true, true),
            &[
                ("  hello  ", "hello"),
                ("\t hi \n", "hi"),
                ("hello", "hello"),
                ("   ", ""),
                ("", ""),
            ],
        );
    }

    #[test]
    fn strip_trims_only_the_left() {
        assert_normalizes(
            &Strip::new(true, false),
            &[
                ("  hello  ", "hello  "),
                ("\t hi \n", "hi \n"),
                ("hello", "hello"),
                ("   ", ""),
                ("", ""),
            ],
        );
    }

    #[test]
    fn strip_trims_only_the_right() {
        assert_normalizes(
            &Strip::new(false, true),
            &[
                ("  hello  ", "  hello"),
                ("\t hi \n", "\t hi"),
                ("hello", "hello"),
                ("   ", ""),
                ("", ""),
            ],
        );
    }
}
