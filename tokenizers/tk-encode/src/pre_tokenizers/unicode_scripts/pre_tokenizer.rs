use std::sync::LazyLock;

use crate::pipeline;
use crate::pre_tokenizers::unicode_scripts::scripts::{get_script, Script};
use crate::tokenizer::{normalizer::Range, PreTokenizedString, PreTokenizer, Result};
use crate::utils::macro_rules_attribute;

#[derive(Clone, Debug, PartialEq, Eq)]
#[macro_rules_attribute(impl_serde_type!)]
pub struct UnicodeScripts;

impl UnicodeScripts {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for UnicodeScripts {
    fn default() -> Self {
        Self::new()
    }
}

// This code exists in the Unigram default IsValidSentencePiece.
// It could be integrated directly within `get_script` but I
// think it's kind of tricky to see those modifications later
// I am guessing release mode will optimize this away anyway.
fn fixed_script(c: char) -> Script {
    let raw_script = get_script(c);
    if c as u32 == 0x30FC {
        Script::Han
    } else if c == ' ' {
        Script::Any
    } else {
        match raw_script {
            Script::Hiragana => Script::Han,
            Script::Katakana => Script::Han,
            script => script,
        }
    }
}

impl PreTokenizer for UnicodeScripts {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        pretokenized.split(|_, normalized| {
            let mut last_script = None;
            let mut offset = 0;
            let mut ranges: Vec<_> = normalized
                .get()
                .chars()
                .filter_map(|c| {
                    let script = Some(fixed_script(c));
                    let result = if script != Some(Script::Any)
                        && last_script != Some(Script::Any)
                        && last_script != script
                    {
                        Some(offset)
                    } else {
                        None
                    };
                    offset += c.len_utf8();
                    if script != Some(Script::Any) {
                        last_script = script;
                    }

                    result
                })
                .collect();
            ranges.push(normalized.get().len());
            Ok(ranges
                .windows(2)
                .map(|item| {
                    normalized
                        .slice(Range::Normalized(item[0]..item[1]))
                        .expect("NormalizedString bad split")
                })
                .collect::<Vec<_>>())
        })
    }
}

static ASCII_CLASS: LazyLock<[Script; 128]> =
    LazyLock::new(|| std::array::from_fn(|b| fixed_script(b as u8 as char)));

impl pipeline::PreTokenizer for UnicodeScripts {
    fn pre_tokenize(&self, text: &str, out: &mut Vec<pipeline::Split>) -> Result<()> {
        let mut start = None;
        let mut last_script = None;

        for (i, ch) in text.char_indices() {
            let script = if ch.is_ascii() {
                ASCII_CLASS[ch as usize]
            } else {
                fixed_script(ch)
            };
            if script == Script::Any {
                continue;
            }
            if last_script.is_none_or(|ls| ls != script) {
                if let Some(s) = start {
                    out.push(pipeline::Split {
                        start: s,
                        end: i as u32,
                    });
                }
                start = Some(i as u32);
            }
            last_script = Some(script);
        }

        if let Some(start) = start {
            out.push(pipeline::Split {
                start,
                end: text.len() as u32,
            });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OffsetReferential;
    use crate::OffsetType;

    #[test]
    fn basic() {
        let pretok = UnicodeScripts {};
        let mut pretokenized = PreTokenizedString::from("どこで生れ。Yes");
        pretok.pre_tokenize(&mut pretokenized).unwrap();
        assert_eq!(
            pretokenized
                .get_splits(OffsetReferential::Normalized, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>(),
            vec![("どこで生れ", (0, 15)), ("。", (15, 18)), ("Yes", (18, 21))]
        );
        assert_eq!(
            pretokenized
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>(),
            vec![("どこで生れ", (0, 15)), ("。", (15, 18)), ("Yes", (18, 21))]
        );
    }

    #[test]
    fn spaces_are_included_in_every_script() {
        let pretok = UnicodeScripts {};
        let mut pretokenized = PreTokenizedString::from("Apples are りんご 林檎");
        pretok.pre_tokenize(&mut pretokenized).unwrap();
        assert_eq!(
            pretokenized
                .get_splits(OffsetReferential::Normalized, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>(),
            vec![("Apples are ", (0, 11)), ("りんご 林檎", (11, 27))]
        );
        assert_eq!(
            pretokenized
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>(),
            vec![("Apples are ", (0, 11)), ("りんご 林檎", (11, 27))]
        );
    }

    fn pretokenize(text: &str) -> Vec<(&str, (u32, u32))> {
        let pretok = UnicodeScripts;
        let mut splits = Vec::new();
        crate::pipeline::PreTokenizer::pre_tokenize(&pretok, text, &mut splits).unwrap();
        splits
            .iter()
            .map(|s| (&text[s.range()], (s.start, s.end)))
            .collect()
    }

    #[test]
    fn pipeline_basic() {
        // same oracle as the legacy `basic` test: kana+kanji collapse to one Han
        // run, the CJK full stop is its own script, then Latin.
        assert_eq!(
            pretokenize("どこで生れ。Yes"),
            vec![("どこで生れ", (0, 15)), ("。", (15, 18)), ("Yes", (18, 21))],
        );
    }

    #[test]
    fn pipeline_spaces_are_neutral() {
        // spaces (`Script::Any`) never trigger a boundary and stick to the run
        // around them; only the Latin -> Japanese change splits.
        assert_eq!(
            pretokenize("Apples are りんご 林檎"),
            vec![("Apples are ", (0, 11)), ("りんご 林檎", (11, 27))],
        );
        // a trailing space stays attached to the preceding run
        assert_eq!(pretokenize("hi 京"), vec![("hi ", (0, 3)), ("京", (3, 6))],);
    }

    #[test]
    fn pipeline_edge_cases() {
        let empty = Vec::<(&str, (u32, u32))>::new();
        assert_eq!(pretokenize(""), empty);
        // all-neutral input produces nothing (no real script ever seen)
        assert_eq!(pretokenize("   "), empty);
        // single script -> one split
        assert_eq!(pretokenize("hello"), vec![("hello", (0, 5))]);
        // leading spaces are dropped (nothing before the first real script)
        assert_eq!(pretokenize(" hi"), vec![("hi", (1, 3))]);
    }

    #[test]
    fn test_unicode_script() {
        assert_eq!(Script::Han, fixed_script('京'));
        assert_eq!(Script::Han, fixed_script('太'));
        assert_eq!(Script::Han, fixed_script('い'));
        assert_eq!(Script::Han, fixed_script('グ'));
        assert_eq!(Script::Han, fixed_script('ー'));
        assert_eq!(Script::Latin, fixed_script('a'));
        assert_eq!(Script::Latin, fixed_script('A'));
        assert_eq!(Script::Common, fixed_script('0'));
        assert_eq!(Script::Common, fixed_script('$'));
        assert_eq!(Script::Common, fixed_script('@'));
        assert_eq!(Script::Common, fixed_script('-'));
        assert_eq!(Script::Any, fixed_script(' '));
    }
}
