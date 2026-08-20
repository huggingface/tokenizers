//! The one piece of post-processor serde that cannot be an attribute on the type.
//!
//! `BertProcessing`, `RobertaProcessing` and the whole template family carry plain derives next to
//! their definitions. What is left is `TemplateProcessing`'s way *in*: `added_single` and
//! `added_pair` are `#[serde(skip)]` because they are *derived* — counted off `single`/`pair` against
//! `special_tokens` — so writing them would be redundant and reading them would let a hand-edited
//! file disagree with itself. A `#[serde(skip)]` field would deserialize to `Default::default()`,
//! i.e. zero, and `added_tokens` is what truncation budgets off — so something has to recount.
//!
//! That something is this helper plus `#[serde(from = ...)]` on the type: a struct that names only
//! the three fields on the wire, converted through [`TemplateProcessing::from_parts`].
//!
//! ## The `"type"` tag is optional for every post-processor
//!
//! `PostProcessorWrapper` is an *untagged* enum in both directions — it has no tagged dispatch at
//! all, unlike `DecoderWrapper` — so a variant that is lenient about the tag will claim a tag-less
//! object, and a variant that requires one will refuse it. For the post-processors that leniency is
//! not an accident to be tolerated but a documented requirement:
//! `post_processor_deserialization_no_type` asserts that a tag-less Bert and a tag-less Roberta both
//! load. Tag-less post-processors are a real shape found in real files, so none of the three uses
//! the one-variant tag-enum trick, and what discriminates them is their *required fields*.
//!
//! The one processor whose tag *is* required is `Sequence`, and it is not here: that type was
//! deleted with the config layer, and a config naming it is read by `tk-serialize`, which matches
//! on the tag directly. Strict either way, because `{"processors":[…]}` with no tag would otherwise
//! be indistinguishable from nothing at all.

use serde::Deserialize;

use super::template::{Template, TemplateProcessing, Tokens};

/// The three fields of a `TemplateProcessing` that reach disk, plus the `"type"` envelope.
///
/// `rename` is what keeps the way *out* saying `TemplateProcessing`; the way in does not care what
/// the tag says, only that the three fields are there.
#[doc(hidden)]
#[derive(Deserialize)]
#[serde(tag = "type", rename = "TemplateProcessing")]
pub struct TemplateProcessingDeserializer {
    single: Template,
    pair: Template,
    special_tokens: Tokens,
}

impl From<TemplateProcessingDeserializer> for TemplateProcessing {
    fn from(t: TemplateProcessingDeserializer) -> Self {
        TemplateProcessing::from_parts(t.single, t.pair, t.special_tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processors::template::{Piece, Sequence, SpecialToken};

    #[test]
    fn piece_serde() {
        for (piece, json) in [
            (
                Piece::Sequence {
                    id: Sequence::A,
                    type_id: 0,
                },
                r#"{"Sequence":{"id":"A","type_id":0}}"#,
            ),
            (
                Piece::Sequence {
                    id: Sequence::B,
                    type_id: 1,
                },
                r#"{"Sequence":{"id":"B","type_id":1}}"#,
            ),
            (
                Piece::SpecialToken {
                    id: "[CLS]".into(),
                    type_id: 0,
                },
                r#"{"SpecialToken":{"id":"[CLS]","type_id":0}}"#,
            ),
        ] {
            assert_eq!(serde_json::to_string(&piece).unwrap(), json);
            assert_eq!(serde_json::from_str::<Piece>(json).unwrap(), piece);
        }
    }

    #[test]
    fn special_token_serde() {
        let simple = SpecialToken::from(("[CLS]", 0));
        let simple_s = r#"{"id":"[CLS]","ids":[0],"tokens":["[CLS]"]}"#;
        assert_eq!(serde_json::to_string(&simple).unwrap(), simple_s);
        assert_eq!(
            serde_json::from_str::<SpecialToken>(simple_s).unwrap(),
            simple
        );

        let complete = SpecialToken::new(
            "[2FR]".into(),
            vec![1, 2, 3],
            vec!["convert".into(), "to".into(), "FR".into()],
        )
        .unwrap();
        let complete_s = r#"{"id":"[2FR]","ids":[1,2,3],"tokens":["convert","to","FR"]}"#;
        assert_eq!(serde_json::to_string(&complete).unwrap(), complete_s);
        assert_eq!(
            serde_json::from_str::<SpecialToken>(complete_s).unwrap(),
            complete
        );
    }

    /// The `ids`/`tokens` length mismatch that `SpecialToken::new` rejects is accepted by the
    /// derive, because a derive builds a struct literal. If someone "fixes" that by routing the way
    /// in through `new`, this is the test that says no.
    #[test]
    fn special_token_length_mismatch_still_loads() {
        let malformed = r#"{"id":"[2FR]","ids":[1,2],"tokens":["convert","to","FR"]}"#;
        let token = serde_json::from_str::<SpecialToken>(malformed).unwrap();
        assert_eq!(token.ids(), &[1, 2]);
        assert_eq!(token.tokens().len(), 3);
    }

    #[test]
    fn template_serde() {
        let template = Template::new(vec![
            Piece::Sequence {
                id: Sequence::A,
                type_id: 0,
            },
            Piece::SpecialToken {
                id: "[CLS]".into(),
                type_id: 0,
            },
        ]);
        let template_s =
            r#"[{"Sequence":{"id":"A","type_id":0}},{"SpecialToken":{"id":"[CLS]","type_id":0}}]"#;
        assert_eq!(serde_json::to_string(&template).unwrap(), template_s);
        assert_eq!(
            serde_json::from_str::<Template>(template_s).unwrap(),
            template
        );
    }

    #[test]
    fn tokens_serde() {
        let tokens = Tokens::from(vec![("[CLS]", 1), ("[SEP]", 0)]);
        let tokens_s = r#"{"[CLS]":{"id":"[CLS]","ids":[1],"tokens":["[CLS]"]},"[SEP]":{"id":"[SEP]","ids":[0],"tokens":["[SEP]"]}}"#;
        assert_eq!(serde_json::to_string(&tokens).unwrap(), tokens_s);
        assert_eq!(serde_json::from_str::<Tokens>(tokens_s).unwrap(), tokens);
    }

    fn get_bert_template() -> TemplateProcessing {
        TemplateProcessing::builder()
            .try_single(vec!["[CLS]", "$0", "[SEP]"])
            .unwrap()
            .try_pair("[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1")
            .unwrap()
            .special_tokens(vec![("[CLS]", 1), ("[SEP]", 0)])
            .build()
            .unwrap()
    }

    #[test]
    fn template_processing_serde() {
        let template = get_bert_template();
        let template_s = "{\
            \"type\":\"TemplateProcessing\",\
            \"single\":[\
                {\"SpecialToken\":{\"id\":\"[CLS]\",\"type_id\":0}},\
                {\"Sequence\":{\"id\":\"A\",\"type_id\":0}},\
                {\"SpecialToken\":{\"id\":\"[SEP]\",\"type_id\":0}}\
            ],\
            \"pair\":[\
                {\"SpecialToken\":{\"id\":\"[CLS]\",\"type_id\":0}},\
                {\"Sequence\":{\"id\":\"A\",\"type_id\":0}},\
                {\"SpecialToken\":{\"id\":\"[SEP]\",\"type_id\":0}},\
                {\"Sequence\":{\"id\":\"B\",\"type_id\":1}},\
                {\"SpecialToken\":{\"id\":\"[SEP]\",\"type_id\":1}}\
            ],\
            \"special_tokens\":{\
                \"[CLS]\":{\
                    \"id\":\"[CLS]\",\"ids\":[1],\"tokens\":[\"[CLS]\"]\
                },\
                \"[SEP]\":{\
                    \"id\":\"[SEP]\",\"ids\":[0],\"tokens\":[\"[SEP]\"]\
                }\
            }}";
        assert_eq!(serde_json::to_string(&template).unwrap(), template_s);
        assert_eq!(
            serde_json::from_str::<TemplateProcessing>(template_s).unwrap(),
            template
        );
    }

    /// `added_single`/`added_pair` never reach disk, so the only thing that can put them back is the
    /// recount in `from_parts`. Deserializing the bert template has to land on the same 2/3 the
    /// builder computed, or `added_tokens` -- which truncation budgets off -- silently reads zero.
    #[test]
    fn added_counts_are_recomputed_on_the_way_in() {
        let json = serde_json::to_string(&get_bert_template()).unwrap();
        assert!(!json.contains("added_single"));
        assert!(!json.contains("added_pair"));

        let back: TemplateProcessing = serde_json::from_str(&json).unwrap();
        assert_eq!(back.get_added_single(), 2);
        assert_eq!(back.get_added_pair(), 3);
    }

    /// The tag is *optional* on the way in for all three processors -- see the module docs. The
    /// tag's *value* is not checked either, which is what lets `TemplateProcessingDeserializer` read
    /// an object tagged `"TemplateProcessing"`.
    #[test]
    fn template_tag_is_optional_and_its_value_unchecked() {
        for json in [
            r#"{"single":[{"Sequence":{"id":"A","type_id":0}}],"pair":[{"Sequence":{"id":"A","type_id":0}},{"Sequence":{"id":"B","type_id":1}}],"special_tokens":{}}"#,
            r#"{"type":"whatever","single":[{"Sequence":{"id":"A","type_id":0}}],"pair":[{"Sequence":{"id":"A","type_id":0}},{"Sequence":{"id":"B","type_id":1}}],"special_tokens":{}}"#,
        ] {
            assert!(
                serde_json::from_str::<TemplateProcessing>(json).is_ok(),
                "should have parsed: {json}"
            );
        }

        // A missing required field is still an error, and that is what discriminates the untagged
        // variants from each other.
        match serde_json::from_str::<TemplateProcessing>(
            r#"{"type":"TemplateProcessing","single":[]}"#,
        ) {
            Err(err) => assert_eq!(err.to_string(), "missing field `pair` at line 1 column 41"),
            _ => panic!("a template with no `pair` must not parse"),
        }
    }
}
