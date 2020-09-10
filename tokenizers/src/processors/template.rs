//! # Template Processing
//!
//! Provides a way to specify templates in order to add the special tokens to each
//! input sequence as relevant.
//!
//! ## Example
//!
//! Let's take `BERT` tokenizer as an example. It uses two special tokens, used to
//! delimitate each sequence. `[CLS]` is always used at the beginning of the first
//! sequence, and `[SEP]` is added at the end of both the first, and the pair
//! sequences. The final result looks like this:
//! - Single sequence: `[CLS] Hello there [SEP]`
//! - Pair sequences: `[CLS] My name is Anthony [SEP] What is my name? [SEP]`
//!
//! So, we can define a [`TemplateProcessing`] that will achieve this result:
//! ```
//! # use tokenizers::processors::template::TemplateProcessing;
//! let template = TemplateProcessing::builder()
//!     // The first sequence has `[CLS]` first, the input, `[SEP]` at the end
//!     .sequence_a(vec!["[CLS]", "$0", "[SEP]"])
//!     // Same as:
//!     .sequence_a("[CLS] $0 [SEP]")
//!
//!     // The pair sequence just has `[SEP]` at the end
//!     .sequence_b(vec!["$1", "[SEP]"])
//!     .sequence_b("$1 [SEP]")
//!
//!     // The list of special tokens used by each sequences
//!     .special_tokens(vec![("[CLS]", 1), ("[SEP]", 0)])
//!     .build()
//!     .unwrap();
//! ```
//!
//! In this example, `$0` and `$1` both represent the input sequences. The number in this
//! identifier is actually the default `type_id` that will be used for each sequence. So,
//! in this case, the first sequence will use `0`, while the pair sequence will use `1`.
//!
//! Note that we are saying the "default" `type_id` because each `SpecialToken` can define
//! its own `type_id` which would override the provided default.
//!
//! **Warning**: You must ensure that you are giving the correct tokens/ids as these will
//! be added to the `Encoding` without any further check. If the given ids correspond to
//! something totally different in a `Tokenizer` using this `PostProcessor`, it might lead
//! to unexpected results.
//!
//! [`TemplateProcessing`]: struct.TemplateProcessing.html
//!
use crate::{Encoding, PostProcessor, Result};
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Represents the different kind of pieces that constitute a template.
/// It can be either the input sequence or a [`SpecialToken`]:
///
/// - The `Sequence` has an associated `type_id` which is used by default
/// for any token inside this sequence. The `Sequence` corresponds to the
/// input sequence given as input of the `PostProcessor`.
///
/// - The `SpecialToken` has an associated `id`. It corresponds to a [`SpecialToken`].
///
/// The easiest way to build a `Piece` is actually buy converting it from a string:
/// ```
/// # use tokenizers::processors::template::Piece;
/// let sequence_with_type_id_0 = Piece::from("$0");
/// let sequence_with_type_id_1 = Piece::from("$1");
/// let special_token_cls = Piece::from("[CLS]");
/// ```
///
/// [`SpecialToken`]: struct.SpecialToken.html
///
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Piece {
    Sequence { type_id: u32 },
    SpecialToken { id: String },
}

impl From<String> for Piece {
    fn from(s: String) -> Self {
        // Try to extract `$(n)?` first
        if s.starts_with('$') {
            let rest = &s['$'.len_utf8()..];

            // If the id is just `$`, we use 0 as type_id
            if rest == "" {
                return Self::Sequence { type_id: 0 };
            }
            // If we can parse a type_id, let's use it
            if let Ok(n) = rest.parse() {
                return Self::Sequence { type_id: n };
            }
        }
        // Must be a SpecialToken otherwise
        Self::SpecialToken { id: s }
    }
}

impl From<&str> for Piece {
    fn from(s: &str) -> Self {
        Piece::from(s.to_owned())
    }
}

/// Represents a bunch of tokens to be used in a template.
/// Usually, special tokens have only one associated id/token but in
/// some cases, it might be interesting to have multiple ids/tokens.
///
/// # Examples
/// ```
/// # use tokenizers::processors::template::SpecialToken;
/// // Simple cases, where a single id/token is necessary:
/// let cls = SpecialToken::from(("[CLS]", 1));
/// let sep = SpecialToken::from((0, "[SEP]")); // The order in the tuple is not important
///
/// // More complex case with multiple values:
/// let complex = SpecialToken::new(
///     "A complex special token:".into(),
///     vec![0, 1, 2, 3, 4],
///     vec![None, Some(1), Some(2), Some(3), None],
///     vec!["A".into(), "complex".into(), "special".into(), "token".into(), ":".into()]
/// ).unwrap();
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpecialToken {
    /// A unique id used to identify this SpecialToken in the template
    id: String,
    /// The list of associated ids
    ids: Vec<u32>,
    /// The list of type_ids. If provided, it will override the default
    /// `type_id` of the sequence.
    type_ids: Vec<Option<u32>>,
    /// The list of associated tokens
    tokens: Vec<String>,
}

impl From<(String, u32)> for SpecialToken {
    fn from(v: (String, u32)) -> Self {
        Self {
            id: v.0.clone(),
            ids: vec![v.1],
            type_ids: vec![None],
            tokens: vec![v.0],
        }
    }
}
impl From<(&str, u32)> for SpecialToken {
    fn from(v: (&str, u32)) -> Self {
        Self::from((v.0.to_owned(), v.1))
    }
}
impl From<(u32, String)> for SpecialToken {
    fn from(v: (u32, String)) -> Self {
        Self::from((v.1, v.0))
    }
}
impl From<(u32, &str)> for SpecialToken {
    fn from(v: (u32, &str)) -> Self {
        Self::from((v.1.to_owned(), v.0))
    }
}

impl SpecialToken {
    pub fn new(
        id: String,
        ids: Vec<u32>,
        type_ids: Vec<Option<u32>>,
        tokens: Vec<String>,
    ) -> Result<Self> {
        if ids.len() != type_ids.len() || ids.len() != tokens.len() {
            Err("SpecialToken: ids, type_ids and tokens must be of the same length".into())
        } else {
            Ok(Self {
                id,
                ids,
                type_ids,
                tokens,
            })
        }
    }
}

/// A Template represents a Vec<[`Piece`]>.
///
/// We can easily build one as follows
/// ```
/// # use tokenizers::processors::template::Template;
/// // By providing a `String` or `&str`, we just split on whitespaces:
/// let template = Template::from("[CLS] $0 [SEP]");
///
/// // By providing pieces directly:
/// let template = Template::from(vec!["[CLS]", "$0", "[SEP]"]);
/// ```
/// Both of these methods give the same result.
///
/// [`Piece`]: enum.Piece.html
///
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Template(Vec<Piece>);

impl<T> From<Vec<T>> for Template
where
    T: Into<Piece>,
{
    fn from(v: Vec<T>) -> Self {
        Self(v.into_iter().map(|p| p.into()).collect())
    }
}

impl From<String> for Template {
    fn from(s: String) -> Self {
        Self::from(s.as_ref())
    }
}

impl From<&str> for Template {
    fn from(s: &str) -> Self {
        Self::from(s.split(' ').collect::<Vec<_>>())
    }
}

/// A bunch of [`SpecialToken`] represented by their ID.
/// Internally, `Tokens` is a `HashMap<String, SpecialToken>` and can be built
/// from a HashMap or a Vec<[`SpecialToken`]>.
///
/// [`SpecialToken`]: struct.SpecialToken.html
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Tokens(pub HashMap<String, SpecialToken>);

impl<T: Into<SpecialToken>> From<Vec<T>> for Tokens {
    fn from(v: Vec<T>) -> Self {
        Self(
            v.into_iter()
                .map(|t| {
                    let token: SpecialToken = t.into();
                    (token.id.clone(), token)
                })
                .collect(),
        )
    }
}

impl From<HashMap<String, SpecialToken>> for Tokens {
    fn from(v: HashMap<String, SpecialToken>) -> Self {
        Self(v)
    }
}

/// This PostProcessor takes care of processing each input `Encoding` by applying
/// the corresponding template, before merging them in the final Encoding.
///
/// A `Template` is actually a sequence of `Piece` that will be
/// concatenated together in the given order. Each `Piece` represents either
/// one of the input `Encoding` or a `SpecialToken`.
///
/// ## Example
/// ```
/// # use tokenizers::processors::template::TemplateProcessing;
/// let template = TemplateProcessing::builder()
///     .sequence_a(vec!["[CLS]", "$0", "[SEP]"])
///     .sequence_b(vec!["$1", "[SEP]"])
///     .special_tokens(vec![("[CLS]", 1), ("[SEP]", 0)])
///     .build()
///     .unwrap();
/// ```
///
#[derive(Debug, Clone, PartialEq, Builder, Serialize, Deserialize)]
#[serde(tag = "type", from = "TemplateProcessingDeserializer")]
#[builder(build_fn(validate = "Self::validate"))]
pub struct TemplateProcessing {
    #[builder(setter(into), default = "self.default_seq(0)")]
    sequence_a: Template,
    #[builder(setter(into), default = "self.default_seq(1)")]
    sequence_b: Template,
    #[builder(setter(skip), default = "self.default_added(true)")]
    #[serde(skip)]
    added_a: usize,
    #[builder(setter(skip), default = "self.default_added(false)")]
    #[serde(skip)]
    added_b: usize,
    #[builder(setter(into), default)]
    special_tokens: Tokens,
}

/// We use this custom deserializer to provided the values for `added_a` and `added_b`
/// during deserialization, while not having to serialize them
#[doc(hidden)]
#[derive(Deserialize)]
#[serde(tag = "type")]
struct TemplateProcessingDeserializer {
    sequence_a: Template,
    sequence_b: Template,
    special_tokens: Tokens,
}
impl From<TemplateProcessingDeserializer> for TemplateProcessing {
    fn from(t: TemplateProcessingDeserializer) -> Self {
        let added_a = count_added(&t.sequence_a, Some(&t.special_tokens));
        let added_b = count_added(&t.sequence_b, Some(&t.special_tokens));
        Self {
            sequence_a: t.sequence_a,
            sequence_b: t.sequence_b,
            added_a,
            added_b,
            special_tokens: t.special_tokens,
        }
    }
}

/// Count the number of added tokens in the given template
fn count_added(container: &Template, special_tokens: Option<&Tokens>) -> usize {
    container
        .0
        .iter()
        .map(|p| match p {
            Piece::Sequence { .. } => 0,
            Piece::SpecialToken { id } => {
                special_tokens.map_or(0, |spt| spt.0.get(id).map_or(0, |s| s.ids.len()))
            }
        })
        .sum()
}

impl TemplateProcessingBuilder {
    fn default_seq(&self, type_id: u32) -> Template {
        Template(vec![Piece::Sequence { type_id }])
    }

    fn default_added(&self, is_a: bool) -> usize {
        let container = if is_a {
            self.sequence_a.as_ref()
        } else {
            self.sequence_b.as_ref()
        };
        container.map_or(0, |pieces| {
            count_added(pieces, self.special_tokens.as_ref())
        })
    }

    fn validate(&self) -> std::result::Result<(), String> {
        let check = |sp| {
            let exist = self
                .special_tokens
                .as_ref()
                .map_or(false, |map| map.0.contains_key(sp));

            match exist {
                false => Some(sp),
                true => None,
            }
        };

        let empty = vec![];
        let missing: HashSet<&str> = self
            .sequence_a
            .as_ref()
            .map_or(empty.iter(), |s| s.0.iter())
            .chain(
                self.sequence_b
                    .as_ref()
                    .map_or(empty.iter(), |s| s.0.iter()),
            )
            .filter_map(|piece| match piece {
                Piece::Sequence { .. } => None,
                Piece::SpecialToken { id } => check(id.as_ref()),
            })
            .collect::<HashSet<_>>();

        if missing.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "Missing SpecialToken(s) with id(s) `{}`",
                missing.iter().join(", ")
            ))
        }
    }
}

impl Default for TemplateProcessing {
    fn default() -> Self {
        Self {
            sequence_a: vec!["$0"].into(),
            sequence_b: vec!["$1"].into(),
            added_a: 0,
            added_b: 0,
            special_tokens: Tokens::default(),
        }
    }
}

impl TemplateProcessing {
    pub fn builder() -> TemplateProcessingBuilder {
        TemplateProcessingBuilder::default()
    }

    fn apply_template(
        &self,
        template: &[Piece],
        mut encoding: Encoding,
        add_special_tokens: bool,
    ) -> Result<Encoding> {
        // Compute the new size
        let mut new_len = 0;
        let mut default_type_id = 0;
        for piece in template {
            new_len += match piece {
                Piece::Sequence { type_id } => {
                    default_type_id = *type_id;
                    encoding.len()
                }
                Piece::SpecialToken { id } => {
                    if add_special_tokens {
                        self.special_tokens
                            .0
                            .get(id)
                            .ok_or_else(|| format!("Missing SpecialToken with id {}", id))?
                            .ids
                            .len()
                    } else {
                        0
                    }
                }
            };
        }

        // Then build the new Encoding
        let mut ids = Vec::with_capacity(new_len);
        let mut type_ids = Vec::with_capacity(new_len);
        let mut tokens = Vec::with_capacity(new_len);
        let mut words = Vec::with_capacity(new_len);
        let mut offsets = Vec::with_capacity(new_len);
        let mut special_tokens_mask = Vec::with_capacity(new_len);
        let mut attention_mask = Vec::with_capacity(new_len);
        let overflowing = encoding
            .take_overflowing()
            .into_iter()
            .map(|encoding| self.apply_template(template, encoding, add_special_tokens))
            .collect::<Result<Vec<_>>>()?;

        for piece in template {
            match piece {
                Piece::Sequence { type_id } => {
                    ids.extend(encoding.get_ids());
                    type_ids.extend(std::iter::repeat(type_id).take(encoding.len()));
                    tokens.extend(encoding.get_tokens().iter().map(|s| s.to_owned()));
                    words.extend(encoding.get_words());
                    offsets.extend(encoding.get_offsets());
                    special_tokens_mask.extend(encoding.get_special_tokens_mask());
                    attention_mask.extend(encoding.get_attention_mask());
                }
                Piece::SpecialToken { id } => {
                    if add_special_tokens {
                        let tok = &self.special_tokens.0[id]; // We already checked existance above
                        let len = tok.ids.len();

                        ids.extend(&tok.ids);
                        type_ids
                            .extend(tok.type_ids.iter().map(|id| id.unwrap_or(default_type_id)));
                        tokens.extend(tok.tokens.clone());
                        words.extend(std::iter::repeat(None).take(len));
                        offsets.extend(std::iter::repeat((0, 0)).take(len));
                        special_tokens_mask.extend(std::iter::repeat(1).take(len));
                        attention_mask.extend(std::iter::repeat(1).take(len));
                    }
                }
            }
        }

        Ok(Encoding::new(
            ids,
            type_ids,
            tokens,
            words,
            offsets,
            special_tokens_mask,
            attention_mask,
            overflowing,
        ))
    }
}

impl PostProcessor for TemplateProcessing {
    fn added_tokens(&self, is_pair: bool) -> usize {
        self.added_a + if is_pair { self.added_b } else { 0 }
    }

    fn process(
        &self,
        encoding: Encoding,
        pair: Option<Encoding>,
        add_special_tokens: bool,
    ) -> Result<Encoding> {
        let sequence_a = self.apply_template(&self.sequence_a.0, encoding, add_special_tokens)?;
        let sequence_b = pair
            .map(|encoding| self.apply_template(&self.sequence_b.0, encoding, add_special_tokens))
            .transpose()?;

        PostProcessor::default_process(sequence_a, sequence_b, add_special_tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn piece_serde() {
        let seq_0 = Piece::Sequence { type_id: 0 };
        let seq_0_s = r#"{"Sequence":{"type_id":0}}"#;
        assert_eq!(serde_json::to_string(&seq_0).unwrap(), seq_0_s);
        assert_eq!(serde_json::from_str::<Piece>(seq_0_s).unwrap(), seq_0);

        let seq_1 = Piece::Sequence { type_id: 1 };
        let seq_1_s = r#"{"Sequence":{"type_id":1}}"#;
        assert_eq!(serde_json::to_string(&seq_1).unwrap(), seq_1_s);
        assert_eq!(serde_json::from_str::<Piece>(seq_1_s).unwrap(), seq_1);

        let spe = Piece::SpecialToken { id: "[CLS]".into() };
        let spe_s = r#"{"SpecialToken":{"id":"[CLS]"}}"#;
        assert_eq!(serde_json::to_string(&spe).unwrap(), spe_s);
        assert_eq!(serde_json::from_str::<Piece>(spe_s).unwrap(), spe);
    }

    #[test]
    fn special_token_serde() {
        let simple = SpecialToken::from(("[CLS]", 0));
        let simple_s = r#"{"id":"[CLS]","ids":[0],"type_ids":[null],"tokens":["[CLS]"]}"#;
        assert_eq!(serde_json::to_string(&simple).unwrap(), simple_s);
        assert_eq!(
            serde_json::from_str::<SpecialToken>(simple_s).unwrap(),
            simple
        );

        let complete = SpecialToken::new(
            "[2FR]".into(),
            vec![1, 2, 3],
            vec![None, Some(2), None],
            vec!["convert".into(), "to".into(), "FR".into()],
        )
        .unwrap();
        let complete_s = r#"{"id":"[2FR]","ids":[1,2,3],"type_ids":[null,2,null],"tokens":["convert","to","FR"]}"#;
        assert_eq!(serde_json::to_string(&complete).unwrap(), complete_s);
        assert_eq!(
            serde_json::from_str::<SpecialToken>(complete_s).unwrap(),
            complete
        );

        let malformed = SpecialToken::new(
            "[2FR]".into(),
            vec![1, 2],
            vec![None, Some(2), None],
            vec!["convert".into(), "to".into(), "FR".into()],
        );
        assert!(malformed.is_err());
        let malformed = SpecialToken::new(
            "[2FR]".into(),
            vec![1, 2, 3],
            vec![],
            vec!["convert".into(), "to".into(), "FR".into()],
        );
        assert!(malformed.is_err());
        let malformed = SpecialToken::new(
            "[2FR]".into(),
            vec![1, 2, 3],
            vec![None, None, None],
            vec!["convert".into(), "FR".into()],
        );
        assert!(malformed.is_err());
    }

    #[test]
    fn template_serde() {
        let template = Template::from(vec![
            Piece::Sequence { type_id: 0 },
            Piece::SpecialToken { id: "[CLS]".into() },
        ]);
        let template_s = r#"[{"Sequence":{"type_id":0}},{"SpecialToken":{"id":"[CLS]"}}]"#;
        assert_eq!(serde_json::to_string(&template).unwrap(), template_s);
        assert_eq!(
            serde_json::from_str::<Template>(template_s).unwrap(),
            template
        );
    }

    #[test]
    fn tokens_serde() {
        let tokens = Tokens::from(vec![("[CLS]", 1), ("[SEP]", 0)]);
        let tokens_s = r#"{"[SEP]":{"id":"[SEP]","ids":[0],"type_ids":[null],"tokens":["[SEP]"]},"[CLS]":{"id":"[CLS]","ids":[1],"type_ids":[null],"tokens":["[CLS]"]}}"#;
        let tokens_s_alt = r#"{"[CLS]":{"id":"[CLS]","ids":[1],"type_ids":[null],"tokens":["[CLS]"]},"[SEP]":{"id":"[SEP]","ids":[0],"type_ids":[null],"tokens":["[SEP]"]}}"#;
        let tokens_ser = serde_json::to_string(&tokens).unwrap();
        assert!(tokens_ser == tokens_s || tokens_ser == tokens_s_alt);
        assert_eq!(serde_json::from_str::<Tokens>(tokens_s).unwrap(), tokens);
        assert_eq!(
            serde_json::from_str::<Tokens>(tokens_s_alt).unwrap(),
            tokens
        );
    }

    fn get_bert_template() -> TemplateProcessing {
        TemplateProcessing::builder()
            .sequence_a(vec!["[CLS]", "$0", "[SEP]"])
            .sequence_b(vec!["$1", "[SEP]"])
            .special_tokens(vec![("[CLS]", 1), ("[SEP]", 0)])
            .build()
            .unwrap()
    }

    #[test]
    fn template_processing_serde() {
        let template = tests::get_bert_template();
        let template_s = "{\
            \"type\":\"TemplateProcessing\",\
            \"sequence_a\":[\
                {\"SpecialToken\":{\"id\":\"[CLS]\"}},\
                {\"Sequence\":{\"type_id\":0}},\
                {\"SpecialToken\":{\"id\":\"[SEP]\"}}\
            ],\
            \"sequence_b\":[\
                {\"Sequence\":{\"type_id\":1}},\
                {\"SpecialToken\":{\"id\":\"[SEP]\"}}\
            ],\
            \"special_tokens\":{\
                \"[CLS]\":{\
                    \"id\":\"[CLS]\",\"ids\":[1],\"type_ids\":[null],\"tokens\":[\"[CLS]\"]\
                },\
                \"[SEP]\":{\
                    \"id\":\"[SEP]\",\"ids\":[0],\"type_ids\":[null],\"tokens\":[\"[SEP]\"]\
                }\
            }}";
        let template_s_alt = "{\
            \"type\":\"TemplateProcessing\",\
            \"sequence_a\":[\
                {\"SpecialToken\":{\"id\":\"[CLS]\"}},\
                {\"Sequence\":{\"type_id\":0}},\
                {\"SpecialToken\":{\"id\":\"[SEP]\"}}\
            ],\
            \"sequence_b\":[\
                {\"Sequence\":{\"type_id\":1}},\
                {\"SpecialToken\":{\"id\":\"[SEP]\"}}\
            ],\
            \"special_tokens\":{\
                \"[SEP]\":{\
                    \"id\":\"[SEP]\",\"ids\":[0],\"type_ids\":[null],\"tokens\":[\"[SEP]\"]\
                },\
                \"[CLS]\":{\
                    \"id\":\"[CLS]\",\"ids\":[1],\"type_ids\":[null],\"tokens\":[\"[CLS]\"]\
                }\
            }}";
        let template_ser = serde_json::to_string(&template).unwrap();
        assert!(template_ser == template_s || template_ser == template_s_alt);
        assert_eq!(
            serde_json::from_str::<TemplateProcessing>(template_s).unwrap(),
            template
        );
        assert_eq!(
            serde_json::from_str::<TemplateProcessing>(template_s_alt).unwrap(),
            template
        );
    }

    #[test]
    fn missing_special_tokens() {
        let processor = TemplateProcessing::builder()
            .sequence_a("[CLS] $0 [SEP]")
            .sequence_b("$1 [SEP]")
            .build();

        let err_a = Err("Missing SpecialToken(s) with id(s) `[SEP], [CLS]`".into());
        let err_b = Err("Missing SpecialToken(s) with id(s) `[CLS], [SEP]`".into());
        assert!(processor == err_a || processor == err_b);
    }

    #[test]
    fn template_processing() {
        let processor = tests::get_bert_template();
        assert_eq!(processor.added_tokens(false), 2);
        assert_eq!(processor.added_tokens(true), 3);

        use crate::Token;
        let encoding = Encoding::from_tokens(
            vec![
                Token::new(12, "Hello".into(), (0, 5)),
                Token::new(14, "there".into(), (6, 11)),
            ],
            0,
        );
        let pair = Encoding::from_tokens(vec![Token::new(15, "pair".into(), (0, 4))], 0);
        assert_eq!(
            processor.process(encoding.clone(), None, true).unwrap(),
            Encoding::new(
                vec![1, 12, 14, 0],
                vec![0, 0, 0, 0],
                vec![
                    "[CLS]".into(),
                    "Hello".into(),
                    "there".into(),
                    "[SEP]".into()
                ],
                vec![None, None, None, None],
                vec![(0, 0), (0, 5), (6, 11), (0, 0)],
                vec![1, 0, 0, 1],
                vec![1, 1, 1, 1],
                vec![]
            )
        );
        assert_eq!(
            processor.process(encoding, Some(pair), true).unwrap(),
            Encoding::new(
                vec![1, 12, 14, 0, 15, 0],
                vec![0, 0, 0, 0, 1, 1],
                vec![
                    "[CLS]".into(),
                    "Hello".into(),
                    "there".into(),
                    "[SEP]".into(),
                    "pair".into(),
                    "[SEP]".into()
                ],
                vec![None, None, None, None, None, None],
                vec![(0, 0), (0, 5), (6, 11), (0, 0), (0, 4), (0, 0)],
                vec![1, 0, 0, 1, 0, 1],
                vec![1, 1, 1, 1, 1, 1],
                vec![]
            )
        );
    }
}
