use crate::pre_tokenizers::unicode_scripts::scripts::{get_script, Script};
use crate::tokenizer::{normalizer::Range, PreTokenizedString, PreTokenizer, Result};

#[derive(Clone, Debug)]
pub struct UnicodeScripts;
impl_serde_unit_struct!(UnicodeScriptsVisitor, UnicodeScripts);

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
                    let result = if last_script != script {
                        Some(offset)
                    } else {
                        None
                    };
                    offset += c.len_utf8();
                    last_script = script;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OffsetReferential;

    #[test]
    fn basic() {
        let pretok = UnicodeScripts::default();
        let mut pretokenized = PreTokenizedString::from("どこで生れ。Yes");
        pretok.pre_tokenize(&mut pretokenized).unwrap();
        assert_eq!(
            pretokenized
                .get_splits(OffsetReferential::Normalized)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>(),
            vec![("どこで生れ", (0, 15)), ("。", (15, 18)), ("Yes", (18, 21))]
        );
        assert_eq!(
            pretokenized
                .get_splits(OffsetReferential::Original)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>(),
            vec![("どこで生れ", (0, 15)), ("。", (15, 18)), ("Yes", (18, 21))]
        );
    }
}
