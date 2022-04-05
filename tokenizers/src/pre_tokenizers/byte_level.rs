use std::collections::{HashMap, HashSet, VecDeque};

use crate::decoders::byte_level::CharKind::{NONE, SYMBOL};
use crate::normalizer::Range;
use crate::NormalizedString;
use onig::Regex;
use serde::{Deserialize, Serialize};

use crate::tokenizer::{
    Decoder, Encoding, PostProcessor, PreTokenizedString, PreTokenizer, Result,
    SplitDelimiterBehavior,
};
use crate::utils::macro_rules_attribute;

fn bytes_char() -> HashMap<u8, char> {
    let mut bs: Vec<u8> = vec![];
    bs.extend(b'!'..=b'~');
    bs.extend(b'\xA1'..=b'\xAC');
    bs.extend(b'\xAE'..=b'\xFF');

    let mut cs: Vec<u32> = bs.iter().map(|i| *i as u32).collect();
    let mut n = 0;

    for b in 0..=255u8 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(u32::pow(2, 8) + n);
            n += 1;
        }
    }

    bs.into_iter()
        .zip(cs)
        .map(|(f, t)| (f, unsafe { std::char::from_u32_unchecked(t) }))
        .collect()
}

lazy_static! {
    static ref RE: Regex =
        Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
            .unwrap();
    static ref BYTES_CHAR: HashMap<u8, char> = bytes_char();
    static ref CHAR_BYTES: HashMap<char, u8> =
        bytes_char().into_iter().map(|(c, b)| (b, c)).collect();
}

#[derive(Copy, Clone, Debug, PartialEq)]
/// Provides all the necessary steps to handle the BPE tokenization at the byte-level. Takes care
/// of all the required processing steps to transform a UTF-8 string as needed before and after the
/// BPE model does its job.
#[macro_rules_attribute(impl_serde_type!)]
#[non_exhaustive]
pub struct ByteLevel {
    /// Whether to add a leading space to the first word. This allows to treat the leading word
    /// just as any other word.
    pub add_prefix_space: bool,
    /// Whether the post processing step should trim offsets to avoid including whitespaces.
    pub trim_offsets: bool,

    /// Whether to use the standard GPT2 regex for whitespace splitting
    /// Set it to False if you want to use your own splitting.
    #[serde(default = "default_true")]
    pub use_regex: bool,
}

fn default_true() -> bool {
    true
}

impl Default for ByteLevel {
    fn default() -> Self {
        Self {
            add_prefix_space: true,
            trim_offsets: true,
            use_regex: true,
        }
    }
}

impl ByteLevel {
    pub fn new(add_prefix_space: bool, trim_offsets: bool, use_regex: bool) -> Self {
        Self {
            add_prefix_space,
            trim_offsets,
            use_regex,
        }
    }

    pub fn alphabet() -> HashSet<char> {
        BYTES_CHAR.values().copied().collect()
    }

    #[must_use]
    pub fn add_prefix_space(mut self, v: bool) -> Self {
        self.add_prefix_space = v;
        self
    }

    #[must_use]
    pub fn trim_offsets(mut self, v: bool) -> Self {
        self.trim_offsets = v;
        self
    }

    #[must_use]
    pub fn use_regex(mut self, v: bool) -> Self {
        self.use_regex = v;
        self
    }
}

#[derive(PartialEq)]
/// Character kind used by the optimized gpt-2 regex logic
enum CharKind {
    LETTER,
    NUMBER,
    SYMBOL,
    WHITESPACE,
    NONE(bool),
}

impl CharKind {
    fn matches(&self, c: &char) -> bool {
        match self {
            CharKind::LETTER => c.is_alphabetic(),
            CharKind::NUMBER => c.is_numeric(),
            SYMBOL => !c.is_numeric() && !c.is_alphabetic() && !c.is_whitespace(),
            CharKind::WHITESPACE => c.is_whitespace(),
            NONE(_) => true,
        }
    }
}

impl From<&char> for CharKind {
    fn from(c: &char) -> Self {
        if c.is_numeric() {
            CharKind::NUMBER
        } else if c.is_alphabetic() {
            CharKind::LETTER
        } else if c.is_whitespace() {
            CharKind::WHITESPACE
        } else {
            CharKind::SYMBOL
        }
    }
}

/// This is an optimized implementation of the GPT-2 REGEX logic
/// Specifically, this pattern:
///   r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
/// The pattern is fairly simple and tokenizes in the following manner:
///  - Is the current token an apostrophe followed by specific characters
///  - Maybe a space followed by letters
///  - Maybe a space followed by numbers
///  - Maybe a space followed by more whitespace
///  - Maybe a space followed by anything other than numbers, letters, and whitespace
fn gpt2_regex_optimized(input: &NormalizedString) -> Result<Vec<NormalizedString>> {
    let mut offset_start = 0usize;
    let mut offset_end = 0usize;
    let mut prev_char_kind: CharKind = CharKind::NONE(false);
    let mut char_queue = VecDeque::with_capacity(4);
    let mut chars = input.chars();
    let mut offsets = Vec::with_capacity(chars.size_hint().1.unwrap_or(input.len()) / 4usize);
    while let Some(c) = char_queue.pop_front().or_else(|| chars.next()) {
        if c == '\'' {
            match prev_char_kind {
                SYMBOL => {
                    offset_end += 1;
                }
                NONE(space) => {
                    if space {
                        offset_end += 1;
                        prev_char_kind = SYMBOL;
                    } else {
                        offset_end += 1;
                        if let Some(next_c) = chars.next() {
                            if next_c == 's' || next_c == 't' || next_c == 'm' || next_c == 'd' {
                                offset_end += 1;
                            } else if next_c == 'r' || next_c == 'v' || next_c == 'l' {
                                if let Some(next_next_c) = chars.next() {
                                    if next_next_c == 'e' || next_next_c == 'l' {
                                        offset_end += 2;
                                    } else {
                                        char_queue.push_back(next_c);
                                        char_queue.push_back(next_next_c);
                                    }
                                } else {
                                    char_queue.push_back(next_c);
                                }
                            } else {
                                char_queue.push_back(next_c);
                            }
                        }
                        offsets.push((offset_start, offset_end));
                        offset_start = offset_end;
                        prev_char_kind = CharKind::NONE(false);
                    }
                }
                _ => {
                    char_queue.push_back(c);
                    offsets.push((offset_start, offset_end));
                    offset_start = offset_end;
                    prev_char_kind = CharKind::NONE(false);
                }
            }
        } else if prev_char_kind.matches(&c) {
            match prev_char_kind {
                CharKind::WHITESPACE => {
                    if c != ' ' {
                        offset_end += 1;
                    } else {
                        if let Some(next_c) = chars.next() {
                            if next_c.is_whitespace() {
                                offset_end += 1;
                                char_queue.push_back(next_c);
                            } else {
                                offsets.push((offset_start, offset_end));
                                offset_start = offset_end;
                                offset_end += 1;
                                char_queue.push_back(next_c);
                                prev_char_kind = NONE(true);
                            }
                        } else {
                            offset_end += 1;
                            offsets.push((offset_start, offset_end));
                            offset_start = offset_end;
                        }
                    }
                }
                CharKind::NONE(prefix_space) => {
                    if !prefix_space && c == ' ' {
                        prev_char_kind = CharKind::NONE(true);
                        offset_end += 1;
                    } else if c == ' ' {
                        if let Some(next_c) = chars.next() {
                            if next_c.is_whitespace() {
                                prev_char_kind = CharKind::WHITESPACE;
                                offset_end += 1;
                                char_queue.push_back(next_c);
                            } else {
                                char_queue.push_back(next_c);
                                offsets.push((offset_start, offset_end));
                                offset_start = offset_end;
                                offset_end += 1;
                                prev_char_kind = CharKind::NONE(true);
                            }
                        } else {
                            offset_end += 1;
                            offsets.push((offset_start, offset_end));
                            offset_start = offset_end;
                        }
                    } else {
                        prev_char_kind = CharKind::from(&c);
                        offset_end += 1;
                    }
                }
                _ => {
                    offset_end += 1;
                }
            }
        } else {
            char_queue.push_back(c);
            offsets.push((offset_start, offset_end));
            offset_start = offset_end;
            prev_char_kind = CharKind::NONE(false);
        }
    }
    if offset_end > offset_start {
        offsets.push((offset_start, offset_end));
    }
    Ok(offsets
        .iter()
        .map(|offset| {
            input
                .slice(Range::Normalized(offset.0..offset.1))
                .expect("NormalizedString bad split")
        })
        .collect())
}

/// As a `PreTokenizer`, `ByteLevel` is in charge of transforming all the unicode characters into
/// their byte-level counterpart. It also splits the input according to the configured regex.
// TODO: Give the ability to modify this regex
impl PreTokenizer for ByteLevel {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        pretokenized.split(|_, mut normalized| {
            if self.add_prefix_space && !normalized.get().starts_with(' ') {
                normalized.prepend(" ");
            }
            if self.use_regex {
                gpt2_regex_optimized(&normalized)
            } else {
                Ok(vec![normalized])
            }
        })?;
        pretokenized.normalize(|normalized| {
            let s = normalized.get();
            let mut transformations: Vec<(char, isize)> = Vec::with_capacity(s.len());
            let mut i = 0;
            for cur_char in s.chars() {
                let size = cur_char.len_utf8();
                let bytes = s[i..i + size].as_bytes();
                i += size;
                transformations.extend(
                    bytes
                        .iter()
                        .enumerate()
                        .map(|(i, b)| (BYTES_CHAR[b], if i > 0 { 1 } else { 0 })),
                );
            }
            normalized.transform(transformations.into_iter(), 0);
            Ok(())
        })
    }
}

/// As a `Decoder`, `ByteLevel` is in charge of converting any byte-level characters to their
/// unicode counterpart, before merging everything back into a single String.
/// This decoder will consume the tokens and merge them in one step to alleviate
/// the fact that single token decoded might be a byte not representable as
/// as String.
impl Decoder for ByteLevel {
    fn decode(&self, tokens: Vec<String>) -> Result<Vec<String>> {
        let toks = tokens
            .into_iter()
            .flat_map(|t| {
                t.chars()
                    .try_fold(vec![], |mut acc, c| {
                        CHAR_BYTES.get(&c).map(|b| {
                            acc.push(*b);
                            acc
                        })
                    })
                    .unwrap_or_else(|| t.as_bytes().to_vec())
            })
            .collect::<Vec<u8>>();
        Ok(vec![String::from_utf8_lossy(&toks).to_string()])
    }
}

/// As a `PostProcessor`, `ByteLevel` is in charge of trimming the offsets if necessary.
impl PostProcessor for ByteLevel {
    fn added_tokens(&self, _is_pair: bool) -> usize {
        0
    }

    fn process(
        &self,
        mut encoding: Encoding,
        mut pair_encoding: Option<Encoding>,
        add_special_tokens: bool,
    ) -> Result<Encoding> {
        if self.trim_offsets {
            process_offsets(&mut encoding, self.add_prefix_space);
            encoding
                .get_overflowing_mut()
                .iter_mut()
                .for_each(|encoding| process_offsets(encoding, self.add_prefix_space));

            if let Some(encoding) = pair_encoding.as_mut() {
                process_offsets(encoding, self.add_prefix_space);
                encoding
                    .get_overflowing_mut()
                    .iter_mut()
                    .for_each(|encoding| process_offsets(encoding, self.add_prefix_space));
            }
        }

        <dyn PostProcessor>::default_process(encoding, pair_encoding, add_special_tokens)
    }
}

pub fn process_offsets(encoding: &mut Encoding, add_prefix_space: bool) {
    encoding.process_tokens_with_offsets_mut(|(i, (token, mut offsets))| {
        let mut leading_spaces = token
            .chars()
            .take_while(|c| *c == BYTES_CHAR[&b' '] || c.is_whitespace())
            .count();
        let trailing_spaces = token
            .chars()
            .rev()
            .take_while(|c| *c == BYTES_CHAR[&b' '] || c.is_whitespace())
            .count();

        if leading_spaces > 0 || trailing_spaces > 0 {
            if leading_spaces > 0 {
                // If user uses `is_pretokenized=True` we might have
                // offsets that might begin at the start of the string but are
                // NOT the first token.
                let is_first = i == 0 || offsets.0 == 0;
                if is_first && add_prefix_space && leading_spaces == 1 {
                    // If we are processing the first pair of offsets, with `add_prefix_space`,
                    // then we shouldn't remove anything we added. If there are more than one
                    // leading spaces though, it means we didn't add them, and they should be
                    // removed.
                    leading_spaces = 0;
                }
                offsets.0 = std::cmp::min(offsets.0 + leading_spaces, offsets.1);
            }
            if trailing_spaces > 0 && offsets.1 >= trailing_spaces {
                offsets.1 = std::cmp::max(offsets.1 - trailing_spaces, offsets.0);
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::{
        Decoder, Encoding, OffsetReferential, OffsetType, PostProcessor, PreTokenizedString,
        PreTokenizer,
    };
    use std::iter::{zip, FromIterator};

    #[test]
    fn pre_tokenization() {
        let bytelevel = ByteLevel::default().add_prefix_space(false);
        let mut pretokenized: PreTokenizedString = "Hello my friend, how is your day going?".into();
        bytelevel.pre_tokenize(&mut pretokenized).unwrap();
        assert_eq!(
            pretokenized
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>(),
            vec![
                ("Hello", (0, 5)),
                ("Ġmy", (5, 8)),
                ("Ġfriend", (8, 15)),
                (",", (15, 16)),
                ("Ġhow", (16, 20)),
                ("Ġis", (20, 23)),
                ("Ġyour", (23, 28)),
                ("Ġday", (28, 32)),
                ("Ġgoing", (32, 38)),
                ("?", (38, 39))
            ]
        );
    }

    #[test]
    fn pre_tokenization_no_regex() {
        let bytelevel = ByteLevel::default().use_regex(false);
        let mut pretokenized: PreTokenizedString = "Hello my friend, how is your day going?".into();
        bytelevel.pre_tokenize(&mut pretokenized).unwrap();
        assert_eq!(
            pretokenized
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>(),
            vec![("ĠHelloĠmyĠfriend,ĠhowĠisĠyourĠdayĠgoing?", (0, 39))]
        );
    }

    #[test]
    fn decoding() {
        let bytelevel = ByteLevel::default().add_prefix_space(false);
        assert_eq!(
            bytelevel
                .decode(
                    vec![
                        "Hello", "Ġmy", "Ġfriend", ",", "Ġhow", "Ġis", "Ġyour", "Ġday", "Ġgoing",
                        "?"
                    ]
                    .into_iter()
                    .map(|s| s.into())
                    .collect::<Vec<String>>()
                )
                .unwrap(),
            vec!["Hello my friend, how is your day going?"]
        );
    }

    #[test]
    fn add_prefix_space() {
        let bytelevel = ByteLevel::default().add_prefix_space(true);
        for s in &[
            " Hello my friend, how is your day going?",
            "Hello my friend, how is your day going?",
        ] {
            let mut pretokenized = PreTokenizedString::from(*s);
            bytelevel.pre_tokenize(&mut pretokenized).unwrap();
            assert_eq!(
                pretokenized
                    .get_splits(OffsetReferential::Normalized, OffsetType::Byte)
                    .into_iter()
                    .map(|(s, o, _)| (s, o))
                    .collect::<Vec<_>>(),
                vec![
                    ("ĠHello", (0, 7)),
                    ("Ġmy", (7, 11)),
                    ("Ġfriend", (11, 19)),
                    (",", (19, 20)),
                    ("Ġhow", (20, 25)),
                    ("Ġis", (25, 29)),
                    ("Ġyour", (29, 35)),
                    ("Ġday", (35, 40)),
                    ("Ġgoing", (40, 47)),
                    ("?", (47, 48))
                ]
            );
        }
    }

    #[test]
    fn decode_works_on_separated_tokens() {
        let samples = vec![
            "A Nuskhuri abbreviation of იესუ ქრისტე ( iesu kriste ) \" Jesus Christ \"",
            "An equal number have descenders , like p or q in English \
                 : გ , დ , ე , ვ , კ , ლ , ჟ , ტ , უ , ფ , ღ , ყ , ც",
        ];

        let bytelevel = ByteLevel::default().add_prefix_space(false);
        for sample in samples {
            let mut pretokenized = PreTokenizedString::from(sample);
            bytelevel.pre_tokenize(&mut pretokenized).unwrap();
            let separated_tokens = pretokenized
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .iter()
                .flat_map(|(s, _, _)| s.split("").map(|t| t.into()))
                .collect::<Vec<_>>();
            assert_eq!(sample, bytelevel.decode(separated_tokens).unwrap().join(""));
        }
    }

    #[test]
    fn handling_of_newlines() {
        let mut pretokenized = PreTokenizedString::from("Hello there\nHello there");
        let bytelevel = ByteLevel::default().add_prefix_space(false);
        bytelevel.pre_tokenize(&mut pretokenized).unwrap();

        assert_eq!(
            pretokenized
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>(),
            vec![
                ("Hello", (0, 5)),
                ("Ġthere", (5, 11)),
                ("Ċ", (11, 12)),
                ("Hello", (12, 17)),
                ("Ġthere", (17, 23))
            ]
        );
    }

    #[test]
    fn handling_of_multiple_whitespaces() {
        let mut pretokenized = PreTokenizedString::from("Hello there       dear");
        let bytelevel = ByteLevel::default().add_prefix_space(false);
        bytelevel.pre_tokenize(&mut pretokenized).unwrap();

        assert_eq!(
            pretokenized
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>(),
            vec![
                ("Hello", (0, 5)),
                ("Ġthere", (5, 11)),
                ("ĠĠĠĠĠĠ", (11, 17)),
                ("Ġdear", (17, 22))
            ]
        );
    }

    #[test]
    fn offsets_when_char_split_up() {
        let input = "i⭢j";
        let mut pretokenized = PreTokenizedString::from(input);
        let bytelevel = ByteLevel::default().add_prefix_space(false);
        bytelevel.pre_tokenize(&mut pretokenized).unwrap();

        assert_eq!(
            pretokenized
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>(),
            vec![("i", (0, 1)), ("âŃ¢", (1, 4)), ("j", (4, 5))]
        );
        assert_eq!(
            pretokenized
                .get_splits(OffsetReferential::Normalized, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>(),
            vec![("i", (0, 1)), ("âŃ¢", (1, 7)), ("j", (7, 8))]
        );
        assert_eq!(
            pretokenized
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(_, o, _)| &input[o.0..o.1])
                .collect::<Vec<_>>(),
            vec!["i", "⭢", "j"]
        );
    }

    #[test]
    fn processor_trims_offsets_pre_tokenized() {
        // If user uses `is_pretokenized=True` we might have
        // offsets that might begin at the start of the string but are
        // NOT the first token.
        let mut encoding = Encoding::new(
            vec![0; 5],
            vec![],
            vec!["Ġl".into(), "ove".into(), "Ġl".into(), "ove".into()],
            vec![],
            vec![(0, 1), (1, 4), (0, 1), (1, 4)],
            vec![],
            vec![],
            vec![],
            HashMap::new(),
        );
        process_offsets(&mut encoding, true);
        assert_eq!(
            encoding,
            Encoding::new(
                vec![0; 5],
                vec![],
                vec!["Ġl".into(), "ove".into(), "Ġl".into(), "ove".into()],
                vec![],
                vec![(0, 1), (1, 4), (0, 1), (1, 4)],
                vec![],
                vec![],
                vec![],
                HashMap::new(),
            )
        );
    }

    #[test]
    fn processor_trims_offsets() {
        let start = Encoding::new(
            vec![0; 5],
            vec![],
            vec![
                "Ġ".into(),
                "ĠĠĠĠHelloĠĠ".into(),
                "ĠĠHello".into(),
                "HelloĠĠ".into(),
                "ĠĠĠĠ".into(),
            ],
            vec![],
            vec![(0, 1), (0, 11), (11, 18), (18, 25), (25, 29)],
            vec![],
            vec![],
            vec![],
            HashMap::new(),
        );
        let expected = Encoding::new(
            vec![0; 5],
            vec![],
            vec![
                "Ġ".into(),
                "ĠĠĠĠHelloĠĠ".into(),
                "ĠĠHello".into(),
                "HelloĠĠ".into(),
                "ĠĠĠĠ".into(),
            ],
            vec![],
            vec![(0, 0), (4, 9), (13, 18), (18, 23), (29, 29)],
            vec![],
            vec![],
            vec![],
            HashMap::new(),
        );

        let bytelevel = ByteLevel::default().trim_offsets(true);
        assert_eq!(
            expected,
            bytelevel.process(start.clone(), None, false).unwrap()
        );

        let mut pair_expected = Encoding::new(
            vec![0; 5],
            vec![],
            vec![
                "Ġ".into(),
                "ĠĠĠĠHelloĠĠ".into(),
                "ĠĠHello".into(),
                "HelloĠĠ".into(),
                "ĠĠĠĠ".into(),
            ],
            vec![],
            vec![(0, 0), (4, 9), (13, 18), (18, 23), (29, 29)],
            vec![],
            vec![],
            vec![],
            HashMap::from_iter(vec![(0, 0..5), (1, 5..10)]),
        );
        pair_expected.merge_with(expected, false);
        assert_eq!(
            pair_expected,
            bytelevel
                .process(start.clone(), Some(start), false)
                .unwrap()
        );
    }

    #[test]
    fn decode_unknown_characters() {
        let byte_level = ByteLevel::default();
        assert_eq!(
            byte_level
                .decode(vec![
                    "Hello".into(),
                    "Ġthere".into(),
                    "Ġdear".into(),
                    "Ġfriend!".into(),
                    "Ġ".into(),
                    "[PA D]".into()
                ])
                .unwrap(),
            vec!["Hello there dear friend! [PA D]"]
        );
    }

    #[test]
    fn deserialization() {
        // Before use_regex
        let byte_level: ByteLevel = serde_json::from_str(
            r#"{"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false}"#,
        )
        .unwrap();
        assert!(byte_level.use_regex);

        // Loading works, new future BC test.
        let byte_level: ByteLevel = serde_json::from_str(
            r#"{"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false, "use_regex": true}"#,
        )
        .unwrap();
        assert!(byte_level.use_regex);

        let byte_level: ByteLevel = serde_json::from_str(
            r#"{"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false, "use_regex": false}"#,
        )
        .unwrap();
        assert!(!byte_level.use_regex);
    }

    #[test]
    fn gpt2_optimized() {
        let re_ref: &Regex = &RE;
        let strings = vec![
            NormalizedString::from(" 'lk 'll it'll 1'll can'tk *('ll#$%^& out ''ll we'lk  This is a    big complicated \n  \n ajs   \n\t\n   hh\nt $%^8 t^ 8oi I'll figure 'lk it out '' we'lk  12#$567UJSDKGhbllasdkjn;;; ; ;"),
            NormalizedString::from(" "),
            NormalizedString::from("  "),
            NormalizedString::from(" \n "),
            NormalizedString::from(""),
            NormalizedString::from("justasingleword"),
            NormalizedString::from(" justasingleword"),
            NormalizedString::from("#$%^&*"),
            NormalizedString::from(" #$%^&*"),
            NormalizedString::from("-23456789"),
            NormalizedString::from(" -23456789"),
            NormalizedString::from("23456789"),
            NormalizedString::from(" 23456789")
            ];
        for s in strings {
            let spl = gpt2_regex_optimized(&s).unwrap();
            let ol_spl = s.split(re_ref, SplitDelimiterBehavior::Isolated).unwrap();
            for (new, old) in zip(spl, ol_spl) {
                assert_eq!(new.get(), old.get());
            }
        }
    }
}
