use std::iter::Peekable;
use std::str::CharIndices;
use crate::tokenizer::{
    PreTokenizedString, 
    PreTokenizer, 
    Result, 
    SplitDelimiterBehavior,
};
use crate::utils::macro_rules_attribute;

/// A pre-tokenizer that splits text into tokens by whitespace while separating
/// word characters (alphanumeric + underscore) from punctuation characters.
/// 
/// This tokenizer groups consecutive word characters together and consecutive
/// punctuation characters together, removing all whitespace in the process.
#[derive(Clone, Debug, PartialEq, Eq)]
#[macro_rules_attribute(impl_serde_type!)]
pub struct Whitespace;

impl Default for Whitespace {
    fn default() -> Self {
        Self
    }
}

// Helper function to check if a character is a word character (alphanumeric or underscore)
fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Helper function to extend the end index while a predicate holds true
fn extend_while<F>(chars: &mut Peekable<CharIndices>, start_idx: usize, mut predicate: F) -> usize
where 
    F: FnMut(char) -> bool,
{
    let mut end_idx = start_idx;
    while let Some(&(next_idx, next_ch)) = chars.peek() {
        if predicate(next_ch) {
            end_idx = next_idx + next_ch.len_utf8();
            chars.next();
        } else {
            break;
        }
    }
    end_idx
}

/// Custom pattern struct that implements the splitting logic manually.
/// 
/// This pattern identifies three types of token spans:
/// - Whitespace sequences (marked for removal)
/// - Word character sequences (marked to keep)
/// - Punctuation sequences (marked to keep)
struct ManualWhitespacePattern;

impl crate::tokenizer::pattern::Pattern for ManualWhitespacePattern {
    fn find_matches(&self, inside: &str) -> crate::tokenizer::Result<Vec<((usize, usize), bool)>> {
        let mut token_spans = Vec::new();
        let mut chars = inside.char_indices().peekable();
        
        while let Some((start_idx, ch)) = chars.next() {
            if ch.is_ascii_whitespace() {
                let end_idx = extend_while(&mut chars, start_idx + ch.len_utf8(), |c| c.is_ascii_whitespace());
                token_spans.push(((start_idx, end_idx), true));
            } else if is_word_char(ch) {
                let end_idx = extend_while(&mut chars, start_idx + ch.len_utf8(), is_word_char);
                token_spans.push(((start_idx, end_idx), false));
            } else {
                let end_idx = extend_while(&mut chars, start_idx + ch.len_utf8(), |c| {
                    !c.is_ascii_whitespace() && !is_word_char(c)
                });
                token_spans.push(((start_idx, end_idx), false));
            }
        }
        
        Ok(token_spans)
    }
}

impl PreTokenizer for Whitespace {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        // Use our custom pattern that manually identifies tokens
        pretokenized.split(|_, normalized| {
            normalized.split(ManualWhitespacePattern, SplitDelimiterBehavior::Removed)
        })
    }
}

/// A simple pre-tokenizer that splits text on whitespace characters only.
/// 
/// Unlike `Whitespace`, this tokenizer does not separate word characters from
/// punctuation - it only splits on whitespace boundaries, keeping punctuation
/// attached to adjacent word characters.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[macro_rules_attribute(impl_serde_type!)]
pub struct WhitespaceSplit;

impl PreTokenizer for WhitespaceSplit {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        pretokenized.split(|_, normalized| {
            normalized.split(char::is_whitespace, SplitDelimiterBehavior::Removed)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{OffsetReferential, OffsetType, PreTokenizer};

    #[test]
    fn basic() {
        let tests = vec![
            (
                "Hey man!",
                vec![("Hey", (0, 3)), ("man", (4, 7)), ("!", (7, 8))],
            ),
            (
                "How are you doing?",
                vec![
                    ("How", (0, 3)),
                    ("are", (4, 7)),
                    ("you", (8, 11)),
                    ("doing", (12, 17)),
                    ("?", (17, 18)),
                ],
            ),
            ("\n", vec![]),
        ];
        let pretok = Whitespace {};
        for (s, res) in tests {
            let mut pretokenized = PreTokenizedString::from(s);
            pretok.pre_tokenize(&mut pretokenized).unwrap();
            assert_eq!(
                pretokenized
                    .get_splits(OffsetReferential::Original, OffsetType::Byte)
                    .into_iter()
                    .map(|(s, o, _)| (s, o))
                    .collect::<Vec<_>>(),
                res
            );
        }
    }

    #[test]
    fn whitespace_split() {
        let tests = vec![
            ("Hey man!", vec![("Hey", (0, 3)), ("man!", (4, 8))]),
            (
                "Hey, man, Good?",
                vec![("Hey,", (0, 4)), ("man,", (5, 9)), ("Good?", (10, 15))],
            ),
        ];
        let pretok = WhitespaceSplit;
        for (s, res) in tests {
            let mut pretokenized = PreTokenizedString::from(s);
            pretok.pre_tokenize(&mut pretokenized).unwrap();
            assert_eq!(
                pretokenized
                    .get_splits(OffsetReferential::Original, OffsetType::Byte)
                    .into_iter()
                    .map(|(s, o, _)| (s, o))
                    .collect::<Vec<_>>(),
                res
            );
        }
    }
}
