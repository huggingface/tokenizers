use std::sync::LazyLock;

use crate::pipeline;
use crate::pre_tokenizers::unicode_scripts::scripts::{Script, get_script};
use crate::tokenizer::Result;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

static BMP_SCRIPT: LazyLock<[Script; 0x10000]> = LazyLock::new(|| {
    std::array::from_fn(|i| char::from_u32(i as u32).map_or(Script::Common, fixed_script))
});

// SAFETY: every offset is one `str::char_indices` yielded, or `text.len()`, and they are pushed in
// increasing order.
unsafe impl pipeline::PreTokenizer for UnicodeScripts {
    fn pre_tokenize(
        &self,
        text: &str,
        _scratch: &mut pipeline::PreTokenizerScratch,
        out: &mut Vec<pipeline::Span>,
    ) -> Result<()> {
        let mut start = None;
        let mut last_script = None;

        for (i, ch) in text.char_indices() {
            let cp = ch as u32;
            let script = if cp < 0x10000 {
                BMP_SCRIPT[cp as usize]
            } else {
                fixed_script(ch)
            };
            if script == Script::Any {
                continue;
            }
            if last_script.is_none_or(|ls| ls != script) {
                if let Some(s) = start {
                    out.push(pipeline::Span {
                        start: s,
                        end: i as u32,
                    });
                }
                // A leading run of `Script::Any` emits no boundary of its own, so
                // anchor the first span at 0 instead of skipping past it.
                start = Some(if start.is_none() { 0 } else { i as u32 });
            }
            last_script = Some(script);
        }

        if let Some(start) = start {
            out.push(pipeline::Span {
                start,
                end: text.len() as u32,
            });
        } else if !text.is_empty() {
            // Every char was `Script::Any`, so keep the whole input as one span.
            out.push(pipeline::Span {
                start: 0,
                end: text.len() as u32,
            });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pretokenize(text: &str) -> Vec<(&str, (u32, u32))> {
        let pretok = UnicodeScripts;
        let mut scratch = pipeline::PreTokenizerScratch::default();
        let mut splits = Vec::new();
        crate::pipeline::PreTokenizer::pre_tokenize(&pretok, text, &mut scratch, &mut splits)
            .unwrap();
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
        // all-neutral input is kept as one split
        assert_eq!(pretokenize("   "), vec![("   ", (0, 3))]);
        // single script -> one split
        assert_eq!(pretokenize("hello"), vec![("hello", (0, 5))]);
        // leading spaces stay attached to the first run
        assert_eq!(pretokenize(" hi"), vec![(" hi", (0, 3))]);
        assert_eq!(pretokenize("  どこ"), vec![("  どこ", (0, 8))]);
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
