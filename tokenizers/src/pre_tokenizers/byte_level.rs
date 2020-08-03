use crate::tokenizer::{
    normalizer::Range, Decoder, Encoding, PostProcessor, PreTokenizedString, PreTokenizer, Result,
};
use onig::Regex;
use serde::ser::SerializeStruct;
use serde::{Deserialize, Serialize, Serializer};
use std::collections::{HashMap, HashSet};

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

#[derive(Deserialize, Copy, Clone, Debug)]
/// Provides all the necessary steps to handle the BPE tokenization at the byte-level. Takes care
/// of all the required processing steps to transform a UTF-8 string as needed before and after the
/// BPE model does its job.
pub struct ByteLevel {
    /// Whether to add a leading space to the first word. This allows to treat the leading word
    /// just as any other word.
    add_prefix_space: bool,
    /// Whether the post processing step should trim offsets to avoid including whitespaces.
    trim_offsets: bool,
}
impl Default for ByteLevel {
    fn default() -> Self {
        Self {
            add_prefix_space: true,
            trim_offsets: true,
        }
    }
}

impl ByteLevel {
    pub fn new(add_prefix_space: bool, trim_offsets: bool) -> Self {
        ByteLevel {
            add_prefix_space,
            trim_offsets,
        }
    }

    pub fn alphabet() -> HashSet<char> {
        BYTES_CHAR.values().copied().collect()
    }

    pub fn add_prefix_space(mut self, v: bool) -> Self {
        self.add_prefix_space = v;
        self
    }

    pub fn trim_offsets(mut self, v: bool) -> Self {
        self.trim_offsets = v;
        self
    }
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

            Ok(RE
                .find_iter(normalized.get())
                .map(|(start, end)| {
                    let mut part = normalized
                        .slice_bytes(Range::Normalized(start..end))
                        .expect("Byte-level cannot split according to regex");

                    let mut transformations: Vec<(char, isize)> =
                        Vec::with_capacity(part.get().len());
                    let mut i = 0;
                    for cur_char in part.get().chars() {
                        let size = cur_char.len_utf8();
                        let bytes = part.get()[i..i + size].as_bytes();
                        i += size;
                        transformations.extend(
                            bytes
                                .iter()
                                .enumerate()
                                .map(|(i, b)| (BYTES_CHAR[b], if i > 0 { 1 } else { 0 })),
                        );
                    }
                    part.transform(transformations.into_iter(), 0);

                    part
                })
                .collect::<Vec<_>>())
        })
    }
}

/// As a `Decoder`, `ByteLevel` is in charge of converting any byte-level characters to their
/// unicode counterpart, before merging everything back into a single String.
impl Decoder for ByteLevel {
    fn decode(&self, tokens: Vec<String>) -> Result<String> {
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
            .collect::<Vec<_>>();
        Ok(String::from_utf8_lossy(&toks).into_owned())
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
                .for_each(|mut encoding| process_offsets(&mut encoding, self.add_prefix_space));

            if let Some(mut encoding) = pair_encoding.as_mut() {
                process_offsets(&mut encoding, self.add_prefix_space);
                encoding
                    .get_overflowing_mut()
                    .iter_mut()
                    .for_each(|mut encoding| process_offsets(&mut encoding, self.add_prefix_space));
            }
        }

        PostProcessor::default_process(encoding, pair_encoding, add_special_tokens)
    }
}

pub fn process_offsets(encoding: &mut Encoding, add_prefix_space: bool) {
    let modifs = encoding
        .get_tokens()
        .iter()
        .map(|token| {
            let leading_spaces = token
                .chars()
                .take_while(|c| *c == BYTES_CHAR[&b' '] || c.is_whitespace())
                .count();
            let trailing_spaces = token
                .chars()
                .rev()
                .take_while(|c| *c == BYTES_CHAR[&b' '] || c.is_whitespace())
                .count();
            (leading_spaces, trailing_spaces)
        })
        .enumerate()
        .filter(|(_, v)| v.0 > 0 || v.1 > 0)
        .collect::<Vec<_>>();

    modifs.into_iter().for_each(|(i, (mut ld, tl))| {
        let mut offsets = &mut encoding.get_offsets_mut()[i];
        if ld > 0 {
            if i == 0 && add_prefix_space && ld == 1 {
                // If we are processing the first pair of offsets, with `add_prefix_space`,
                // then we shouldn't remove anything we added. If there are more than one
                // leading spaces though, it means we didn't add them, and they should be
                // removed.
                ld = 0;
            }
            offsets.0 = std::cmp::min(offsets.0 + ld, offsets.1);
        }
        if tl > 0 && offsets.1 >= tl {
            offsets.1 = std::cmp::max(offsets.1 - tl, offsets.0);
        }
    });
}

impl Serialize for ByteLevel {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut m = serializer.serialize_struct("ByteLevel", 3)?;
        m.serialize_field("type", "ByteLevel")?;
        m.serialize_field("add_prefix_space", &self.add_prefix_space)?;
        m.serialize_field("trim_offsets", &self.trim_offsets)?;
        m.end()
    }
}

#[cfg(test)]
mod tests {
    use super::ByteLevel;
    use crate::tokenizer::{
        normalizer::Range, Decoder, Encoding, NormalizedString, OffsetReferential, PostProcessor,
        PreTokenizedString, PreTokenizer,
    };

    #[test]
    fn pre_tokenization() {
        let bytelevel = ByteLevel::default().add_prefix_space(false);
        let mut pretokenized: PreTokenizedString = "Hello my friend, how is your day going?".into();
        bytelevel.pre_tokenize(&mut pretokenized).unwrap();
        assert_eq!(
            pretokenized.get_normalized(OffsetReferential::Original),
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
    fn decoding() {
        let bytelevel = ByteLevel::default().add_prefix_space(false);
        assert_eq!(
            "Hello my friend, how is your day going?",
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
                .unwrap()
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
                pretokenized.get_normalized(OffsetReferential::Normalized),
                vec![
                    ("ĠHello", (0, 6)),
                    ("Ġmy", (6, 9)),
                    ("Ġfriend", (9, 16)),
                    (",", (16, 17)),
                    ("Ġhow", (17, 21)),
                    ("Ġis", (21, 24)),
                    ("Ġyour", (24, 29)),
                    ("Ġday", (29, 33)),
                    ("Ġgoing", (33, 39)),
                    ("?", (39, 40))
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
                .iter()
                .flat_map(|sub| sub.normalized.get().split("").map(|t| t.into()))
                .collect::<Vec<_>>();
            assert_eq!(sample, bytelevel.decode(separated_tokens).unwrap());
        }
    }

    #[test]
    fn handling_of_newlines() {
        let mut pretokenized = PreTokenizedString::from("Hello there\nHello there");
        let bytelevel = ByteLevel::default().add_prefix_space(false);
        bytelevel.pre_tokenize(&mut pretokenized).unwrap();

        assert_eq!(
            pretokenized.get_normalized(OffsetReferential::Original),
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
            pretokenized.get_normalized(OffsetReferential::Original),
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
        let mut pretokenized = PreTokenizedString::from("i⭢j");
        let bytelevel = ByteLevel::default().add_prefix_space(false);
        bytelevel.pre_tokenize(&mut pretokenized).unwrap();

        assert_eq!(
            pretokenized.get_normalized(OffsetReferential::Original),
            vec![("i", (0, 1)), ("âŃ¢", (1, 2)), ("j", (2, 3))]
        );
        assert_eq!(
            pretokenized.get_normalized(OffsetReferential::Normalized),
            vec![("i", (0, 1)), ("âŃ¢", (1, 4)), ("j", (4, 5))]
        );
        assert_eq!(
            pretokenized
                .iter()
                .map(|sub| sub.normalized.get())
                .collect::<String>(),
            "iâŃ¢j"
        );
        assert_eq!(
            pretokenized
                .into_iter()
                .map(|sub| sub.normalized)
                .collect::<NormalizedString>()
                .get_range_original(Range::Normalized(1..4)),
            Some("⭢")
        );
    }

    #[test]
    fn processor_trims_offsets() {
        let start = Encoding::new(
            vec![],
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
        );
        let expected = Encoding::new(
            vec![],
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
        );

        let bytelevel = ByteLevel::default().trim_offsets(true);
        assert_eq!(
            expected,
            bytelevel.process(start.clone(), None, false).unwrap()
        );

        let mut pair_expected = expected.clone();
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
            "Hello there dear friend! [PA D]"
        );
    }
}
