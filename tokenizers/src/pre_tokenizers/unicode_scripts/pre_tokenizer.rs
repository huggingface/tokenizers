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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OffsetReferential;
    use crate::OffsetType;

    #[test]
    fn basic() {
        let pretok = UnicodeScripts::default();
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
        let pretok = UnicodeScripts::default();
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
