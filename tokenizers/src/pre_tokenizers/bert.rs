use crate::tokenizer::{PreTokenizer, Result};
use std::collections::HashSet;
use unicode_categories::UnicodeCategories;
use unicode_normalization::UnicodeNormalization;

/// Extremely simple tokenization on whitespaces
fn whitespace_tokenize(s: &str) -> Vec<&str> {
    s.trim()
        .split(char::is_whitespace)
        .filter(|s| *s != " ")
        .collect()
}

/// Checks whether a character is whitespace
fn is_whitespace(c: char) -> bool {
    // These are technically control characters but we count them as whitespace
    if c == '\t' || c == '\n' || c == '\r' {
        true
    } else {
        c.is_whitespace()
    }
}

/// Checks whether a character is a control character
fn is_control(c: char) -> bool {
    // These are technically control characters but we count them as whitespace
    if c == '\t' || c == '\n' || c == '\r' {
        false
    } else {
        // The definition of `is_control` here is quite large and contains also
        // Cc, Cf, Cn or Co
        // cf. https://unicode.org/reports/tr44/ (Table 12)
        c.is_other()
    }
}

/// Checks whether a character is chinese
/// This defines a "chinese character" as anything in the CJK Unicode block:
///   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
///
/// Note that the CJK Unicode block is NOT all Japanese and Korean characters,
/// despite its name. The modern Korean Hangul alphabet is a different block,
/// as is Japanese Hiragana and Katakana. Those alphabets are used to write
/// space-separated words, so they are not treated specially and handled
/// like for all of the other languages.
fn is_chinese_char(c: char) -> bool {
    match c as usize {
        0x4E00..=0x9FFF => true,
        0x3400..=0x4DBF => true,
        0x20000..=0x2A6DF => true,
        0x2A700..=0x2B73F => true,
        0x2B740..=0x2B81F => true,
        0x2B920..=0x2CEAF => true,
        0xF900..=0xFAFF => true,
        0x2F800..=0x2FA1F => true,
        _ => false,
    }
}

pub struct BertPreTokenizer {
    /// Whether to do the basic tokenization
    do_basic_tokenize: bool,
    /// Whether to lower case the input.
    do_lower_case: bool,
    /// A list of token not to split.
    never_split: HashSet<String>,
    /// Whether to tokenize Chinese characters
    tokenize_chinese_chars: bool,
}

impl BertPreTokenizer {
    pub fn new(
        do_basic_tokenize: bool,
        do_lower_case: bool,
        never_split: HashSet<String>,
        tokenize_chinese_chars: bool,
    ) -> Self {
        BertPreTokenizer {
            do_basic_tokenize,
            do_lower_case,
            never_split,
            tokenize_chinese_chars,
        }
    }

    /// Strips accents from a piece of text
    fn run_strip_accents(&self, text: &str) -> String {
        text.nfd()
            .filter(|c| !c.is_mark_nonspacing())
            .collect::<String>()
    }

    /// Splits punctuation on a piece of text.
    fn run_split_on_punc(&self, text: &str) -> Vec<String> {
        if self.never_split.contains(text) {
            return vec![text.to_owned()];
        }

        let mut output: Vec<Vec<char>> = vec![];
        let mut start_new_word = true;
        text.chars().for_each(|c| {
            if c.is_ascii_punctuation() {
                output.push(vec![c]);
                start_new_word = true;
            } else {
                if start_new_word {
                    output.push(vec![]);
                }
                start_new_word = false;
                output.last_mut().unwrap().push(c);
            }
        });

        output
            .into_iter()
            .map(|cs| cs.into_iter().collect::<String>())
            .collect()
    }

    fn tokenize_chinese_chars(&self, text: &str) -> String {
        text.chars()
            .map(|c| {
                if is_chinese_char(c) {
                    vec![' ', c, ' ']
                } else {
                    vec![c]
                }
            })
            .flatten()
            .collect::<String>()
    }

    fn clean_text(&self, text: &str) -> String {
        text.chars()
            .map(|c| {
                if c as usize == 0 || c as usize == 0xfffd || is_control(c) {
                    None
                } else if is_whitespace(c) {
                    Some(' ')
                } else {
                    Some(c)
                }
            })
            .filter(|c| c.is_some())
            .map(|c| c.unwrap())
            .collect::<String>()
    }
}

impl PreTokenizer for BertPreTokenizer {
    fn pre_tokenize(&self, s: &str) -> Result<Vec<String>> {
        if !self.do_basic_tokenize {
            Ok(whitespace_tokenize(&s)
                .into_iter()
                .map(|s| s.to_owned())
                .collect())
        } else {
            let mut text = self.clean_text(s);

            // This was added on November 1st, 2018 for the multilingual and Chinese
            // models. This is also applied to the English models now, but it doesn't
            // matter since the English models were not trained on any Chinese data
            // and generally don't have any Chinese data in them (there are Chinese
            // characters in the vocabulary because Wikipedia does have some Chinese
            // words in the English Wikipedia.).
            if self.tokenize_chinese_chars {
                text = self.tokenize_chinese_chars(&text);
            }
            let orig_tokens = whitespace_tokenize(&text);
            let mut split_tokens = vec![];
            for token in orig_tokens {
                let mut tk = token.to_owned();
                if self.do_lower_case && !self.never_split.contains(token) {
                    tk = self.run_strip_accents(&token.to_lowercase())
                }
                split_tokens.extend(self.run_split_on_punc(&tk));
            }

            Ok(split_tokens)
        }
    }
}
