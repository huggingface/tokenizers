//! serde for every post-processor: `tk-encode` defines the processor types and serializes none of
//! them.
//!
//! Same orphan-rule reason as `decoders::mirror`: `Serialize`/`Deserialize` are foreign traits and
//! `BertProcessing`/`RobertaProcessing`/`TemplateProcessing` are foreign types, so this crate cannot
//! implement one for the other. It declares a local mirror of the JSON shape and converts, which is
//! what `#[serde(with = "...")]` on `PostProcessorWrapper`'s variants asks for.
//!
//! ## Two mirror shapes, and how to pick
//!
//! * **`remote`** — serde drives the foreign type directly, so the codegen (and therefore the
//!   accepted input) is bit-for-bit what the type used to derive itself. Only usable when every
//!   field is `pub` *and* the type is not `#[non_exhaustive]`, because the generated `Deserialize`
//!   builds the type with a struct literal. `BertProcessing` and `RobertaProcessing` both qualify:
//!   plain data, all fields `pub`, no invariant between them.
//! * **explicit local mirrors converted by hand** — everything in `template.rs`. Not because of
//!   private fields alone (though `SpecialToken` has three of them and `TemplateProcessing` has two
//!   derived ones), but because the template types *nest*: `Template` is a `Vec<Piece>` and `Tokens`
//!   is an `AHashMap<String, SpecialToken>`, and `#[serde(with = ...)]` does not reach through a
//!   `Vec` or a map. A `remote` `PieceDef` would need a per-element newtype wrapper at every level;
//!   owned local mirrors compose through the containers for free.
//!
//! ## Whether the `"type"` tag is required is per-processor, and it is load-bearing
//!
//! `PostProcessorWrapper` is an *untagged* enum in both directions — it has no tagged dispatch at
//! all, unlike `DecoderWrapper` — so a variant that is lenient about the tag will claim a tag-less
//! object, and a variant that requires one will refuse it. As `crate::macros` spells out, a bare
//! `#[serde(tag = "type")]` on a struct does **not** make the tag required; it only adds it on the
//! way out.
//!
//! For the post-processors the tag is **optional everywhere**, and unlike the decoders that is not
//! an accident to be tolerated but a documented requirement:
//! `processors::tests::post_processor_deserialization_no_type` asserts that
//! `{"sep":["[SEP]",102],"cls":["[CLS]",101]}` loads as a `Bert` and that
//! `{"sep":["</s>",2],"cls":["<s>",0],"trim_offsets":true,"add_prefix_space":true}` loads as a
//! `Roberta`. Tag-less post-processors are a real shape found in real files. So none of the three
//! mirrors here uses the one-variant tag-enum trick from `decoders::mirror::fuse`; each carries the
//! identical bare `#[serde(tag = "type")]` the type used to carry, and what discriminates the
//! variants is the set of *required fields* — which is also why `Roberta` has to stay ahead of
//! `Bert` in the enum, since a Roberta object satisfies Bert's shape but not the other way round.
//!
//! The one processor whose tag *is* required is `Sequence`, and it is not mirrored here: it is this
//! crate's own type, built by `impl_serde_type!`. It has to be strict, because `{"processors":[…]}`
//! with no tag would otherwise be indistinguishable from nothing at all.

use ahash::AHashMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use tk_encode::processors::bert::BertProcessing;
use tk_encode::processors::roberta::RobertaProcessing;
use tk_encode::processors::template::{
    Piece, Sequence, SpecialToken, Template, TemplateProcessing, Tokens,
};

// -------------------------------------------------------------------------------------------------
// BertProcessing
// -------------------------------------------------------------------------------------------------

/// Both fields are `pub` tuples of plain data, so serde's `remote` derive drives the foreign type
/// and no hand-written conversion is needed.
///
/// `rename` is not cosmetic: with `remote` the struct serde sees is called `BertProcessingDef`, and
/// `#[serde(tag = "type")]` takes the tag's *value* from the container name. Without the rename this
/// would write `"type":"BertProcessingDef"` into every saved `tokenizer.json`.
#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename = "BertProcessing", remote = "BertProcessing")]
pub struct BertProcessingDef {
    pub sep: (String, u32),
    pub cls: (String, u32),
}

// -------------------------------------------------------------------------------------------------
// RobertaProcessing
// -------------------------------------------------------------------------------------------------

/// Same `remote` treatment as Bert. Note there is no `#[serde(default)]` on `trim_offsets` or
/// `add_prefix_space`: all four fields are required, and that is the whole reason `Roberta` can be
/// told apart from `Bert` by an untagged enum.
#[derive(Serialize, Deserialize)]
#[serde(
    tag = "type",
    rename = "RobertaProcessing",
    remote = "RobertaProcessing"
)]
pub struct RobertaProcessingDef {
    pub sep: (String, u32),
    pub cls: (String, u32),
    pub trim_offsets: bool,
    pub add_prefix_space: bool,
}

// -------------------------------------------------------------------------------------------------
// Sequence / Piece — the leaves of a template
// -------------------------------------------------------------------------------------------------

/// Which of the two input sequences a [`Piece`] refers to, written as the bare variant name.
#[derive(Serialize, Deserialize)]
pub enum SequenceMirror {
    A,
    B,
}

impl From<&Sequence> for SequenceMirror {
    fn from(s: &Sequence) -> Self {
        match s {
            Sequence::A => Self::A,
            Sequence::B => Self::B,
        }
    }
}

impl From<SequenceMirror> for Sequence {
    fn from(m: SequenceMirror) -> Self {
        match m {
            SequenceMirror::A => Self::A,
            SequenceMirror::B => Self::B,
        }
    }
}

/// Externally tagged, which is what the plain derive on `Piece` produced:
/// `{"Sequence":{"id":"A","type_id":0}}` and `{"SpecialToken":{"id":"[CLS]","type_id":0}}`.
///
/// `Piece` also has a `TryFrom<&str>` used all over the builder API (`"[CLS]:0"`, `"$A:1"`, `"$0"`),
/// and it is worth being explicit that it plays **no** part in the JSON shape. A template written as
/// a *string* is parsed by that `TryFrom`; a template read from a `tokenizer.json` is always this
/// object form. The string spelling never reaches disk.
#[derive(Serialize, Deserialize)]
pub enum PieceMirror {
    Sequence { id: SequenceMirror, type_id: u32 },
    SpecialToken { id: String, type_id: u32 },
}

impl From<&Piece> for PieceMirror {
    fn from(p: &Piece) -> Self {
        match p {
            Piece::Sequence { id, type_id } => Self::Sequence {
                id: id.into(),
                type_id: *type_id,
            },
            Piece::SpecialToken { id, type_id } => Self::SpecialToken {
                id: id.clone(),
                type_id: *type_id,
            },
        }
    }
}

impl From<PieceMirror> for Piece {
    fn from(m: PieceMirror) -> Self {
        match m {
            PieceMirror::Sequence { id, type_id } => Self::Sequence {
                id: id.into(),
                type_id,
            },
            PieceMirror::SpecialToken { id, type_id } => Self::SpecialToken { id, type_id },
        }
    }
}

// -------------------------------------------------------------------------------------------------
// SpecialToken
// -------------------------------------------------------------------------------------------------

/// The three fields of a `SpecialToken`, in declaration order.
///
/// The conversion back goes through [`SpecialToken::from_parts`] and **not** through
/// `SpecialToken::new`, deliberately. `new` rejects `ids.len() != tokens.len()`; the `Deserialize`
/// derive this replaces did not, because a derive builds a struct literal. A `tokenizer.json` with
/// three `ids` and two `tokens` therefore loads today and only misbehaves later, when the template
/// is applied — and it has to keep loading. Routing through `new` would be a strictly nicer library
/// and a different one.
#[derive(Serialize, Deserialize)]
pub struct SpecialTokenMirror {
    id: String,
    ids: Vec<u32>,
    tokens: Vec<String>,
}

impl From<&SpecialToken> for SpecialTokenMirror {
    fn from(t: &SpecialToken) -> Self {
        Self {
            id: t.id().to_owned(),
            ids: t.ids().to_vec(),
            tokens: t.tokens().to_vec(),
        }
    }
}

impl From<SpecialTokenMirror> for SpecialToken {
    fn from(m: SpecialTokenMirror) -> Self {
        SpecialToken::from_parts(m.id, m.ids, m.tokens)
    }
}

// -------------------------------------------------------------------------------------------------
// Template / Tokens — the two containers
// -------------------------------------------------------------------------------------------------

/// `Template` was `#[serde(transparent)]` over its `Vec<Piece>`, so on disk it is a bare JSON array
/// with no envelope of its own. Kept transparent here for the same reason.
#[derive(Serialize, Deserialize)]
#[serde(transparent)]
pub struct TemplateMirror(Vec<PieceMirror>);

impl From<&Template> for TemplateMirror {
    fn from(t: &Template) -> Self {
        Self(t.as_slice().iter().map(PieceMirror::from).collect())
    }
}

impl From<TemplateMirror> for Template {
    fn from(m: TemplateMirror) -> Self {
        Template::new(m.0.into_iter().map(Piece::from).collect())
    }
}

/// `Tokens` was `#[serde(transparent)]` over its map, with `ordered_map` on the way out.
///
/// The `serialize_with` is load-bearing and not a nicety: the field is an `AHashMap`, whose
/// iteration order is unspecified, so without it two saves of the same tokenizer would emit
/// `special_tokens` in different orders. `crate::mirror::ordered_map` sorts through a `BTreeMap`.
/// The way *in* is a plain map, as it always was — key order is irrelevant when reading.
#[derive(Serialize, Deserialize)]
#[serde(transparent)]
pub struct TokensMirror(
    #[serde(serialize_with = "crate::mirror::ordered_map")] AHashMap<String, SpecialTokenMirror>,
);

impl From<&Tokens> for TokensMirror {
    fn from(t: &Tokens) -> Self {
        Self(
            t.0.iter()
                .map(|(k, v)| (k.clone(), SpecialTokenMirror::from(v)))
                .collect(),
        )
    }
}

impl From<TokensMirror> for Tokens {
    fn from(m: TokensMirror) -> Self {
        Tokens::from(
            m.0.into_iter()
                .map(|(k, v)| (k, SpecialToken::from(v)))
                .collect::<AHashMap<_, _>>(),
        )
    }
}

// -------------------------------------------------------------------------------------------------
// TemplateProcessing
// -------------------------------------------------------------------------------------------------

/// The three fields of a `TemplateProcessing` that reach disk, plus the `"type"` envelope.
///
/// `TemplateProcessing` itself has five fields; `added_single` and `added_pair` carried
/// `#[serde(skip)]` because they are *derived* — counted off `single`/`pair` against
/// `special_tokens` — so writing them would be redundant and reading them would let a hand-edited
/// file disagree with itself. The old code spelled that with a `#[doc(hidden)]`
/// `TemplateProcessingDeserializer` helper struct plus `#[serde(from = "…")]`: a separate type that
/// existed only to provide the values for `added_single` and `added_pair` during deserialization
/// while not having to serialize them. One mirror does both jobs here, and the recount happens in
/// [`TemplateProcessing::from_parts`], which is that helper's `From` impl moved into `tk-encode`.
///
/// Two things about the old spelling that this has to reproduce and that are easy to get wrong:
///
/// * The serialized tag was `"TemplateProcessing"` (from `#[serde(tag = "type")]` on
///   `TemplateProcessing`), while the type serde actually *deserialized* was called
///   `TemplateProcessingDeserializer`. That mismatch was harmless only because serde ignores an
///   internally-tagged struct's tag *value* on the way in. Hence `rename` here: the way out must
///   keep saying `TemplateProcessing`, and the way in keeps not caring.
/// * All three fields are required — no `#[serde(default)]` anywhere — which is what stops a
///   `{"sep":…,"cls":…}` Bert object from being read as an empty template by the untagged enum.
#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename = "TemplateProcessing")]
pub struct TemplateProcessingMirror {
    single: TemplateMirror,
    pair: TemplateMirror,
    special_tokens: TokensMirror,
}

impl From<&TemplateProcessing> for TemplateProcessingMirror {
    fn from(t: &TemplateProcessing) -> Self {
        Self {
            single: (&t.single).into(),
            pair: t.get_pair().into(),
            special_tokens: t.get_special_tokens().into(),
        }
    }
}

impl From<TemplateProcessingMirror> for TemplateProcessing {
    fn from(m: TemplateProcessingMirror) -> Self {
        TemplateProcessing::from_parts(m.single.into(), m.pair.into(), m.special_tokens.into())
    }
}

pub mod template {
    use super::*;

    pub fn serialize<S: Serializer>(v: &TemplateProcessing, s: S) -> Result<S::Ok, S::Error> {
        TemplateProcessingMirror::from(v).serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<TemplateProcessing, D::Error> {
        Ok(TemplateProcessingMirror::deserialize(d)?.into())
    }
}

// Every test in here is a JSON round-trip, so they moved out of `tk-encode` with the serde they
// exercise. The expected strings are the ones those tests asserted, character for character.
#[cfg(test)]
mod tests {
    use super::*;

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
            let ser = serde_json::to_string(&PieceMirror::from(&piece)).unwrap();
            assert_eq!(ser, json);
            let back: Piece = serde_json::from_str::<PieceMirror>(json).unwrap().into();
            assert_eq!(back, piece);
        }
    }

    #[test]
    fn special_token_serde() {
        let simple = SpecialToken::from(("[CLS]", 0));
        let simple_s = r#"{"id":"[CLS]","ids":[0],"tokens":["[CLS]"]}"#;
        assert_eq!(
            serde_json::to_string(&SpecialTokenMirror::from(&simple)).unwrap(),
            simple_s
        );
        assert_eq!(
            SpecialToken::from(serde_json::from_str::<SpecialTokenMirror>(simple_s).unwrap()),
            simple
        );

        let complete = SpecialToken::new(
            "[2FR]".into(),
            vec![1, 2, 3],
            vec!["convert".into(), "to".into(), "FR".into()],
        )
        .unwrap();
        let complete_s = r#"{"id":"[2FR]","ids":[1,2,3],"tokens":["convert","to","FR"]}"#;
        assert_eq!(
            serde_json::to_string(&SpecialTokenMirror::from(&complete)).unwrap(),
            complete_s
        );
        assert_eq!(
            SpecialToken::from(serde_json::from_str::<SpecialTokenMirror>(complete_s).unwrap()),
            complete
        );
    }

    /// The `ids`/`tokens` length mismatch that `SpecialToken::new` rejects is accepted here, exactly
    /// as the old `Deserialize` derive accepted it. This is the behaviour `from_parts` exists to
    /// preserve; if someone "fixes" the mirror to go through `new`, this is the test that says no.
    #[test]
    fn special_token_length_mismatch_still_loads() {
        let malformed = r#"{"id":"[2FR]","ids":[1,2],"tokens":["convert","to","FR"]}"#;
        let token =
            SpecialToken::from(serde_json::from_str::<SpecialTokenMirror>(malformed).unwrap());
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
        assert_eq!(
            serde_json::to_string(&TemplateMirror::from(&template)).unwrap(),
            template_s
        );
        assert_eq!(
            Template::from(serde_json::from_str::<TemplateMirror>(template_s).unwrap()),
            template
        );
    }

    #[test]
    fn tokens_serde() {
        let tokens = Tokens::from(vec![("[CLS]", 1), ("[SEP]", 0)]);
        let tokens_s = r#"{"[CLS]":{"id":"[CLS]","ids":[1],"tokens":["[CLS]"]},"[SEP]":{"id":"[SEP]","ids":[0],"tokens":["[SEP]"]}}"#;
        let tokens_ser = serde_json::to_string(&TokensMirror::from(&tokens)).unwrap();
        assert_eq!(tokens_ser, tokens_s);
        assert_eq!(
            Tokens::from(serde_json::from_str::<TokensMirror>(tokens_s).unwrap()),
            tokens
        );
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
        let template_ser =
            serde_json::to_string(&TemplateProcessingMirror::from(&template)).unwrap();
        assert_eq!(template_ser, template_s);
        assert_eq!(
            TemplateProcessing::from(
                serde_json::from_str::<TemplateProcessingMirror>(template_s).unwrap()
            ),
            template
        );
    }

    /// `added_single`/`added_pair` never reach disk, so the only thing that can put them back is the
    /// recount in `from_parts`. Deserializing the bert template has to land on the same 2/3 the
    /// builder computed, or `added_tokens` — which truncation budgets off — silently reads zero.
    #[test]
    fn added_counts_are_recomputed_on_the_way_in() {
        let json =
            serde_json::to_string(&TemplateProcessingMirror::from(&get_bert_template())).unwrap();
        assert!(!json.contains("added_single"));
        assert!(!json.contains("added_pair"));

        let back = TemplateProcessing::from(
            serde_json::from_str::<TemplateProcessingMirror>(&json).unwrap(),
        );
        assert_eq!(back.get_added_single(), 2);
        assert_eq!(back.get_added_pair(), 3);
    }

    /// The tag is *optional* on the way in for all three processors — see the module docs. The tag's
    /// value is not checked either, which is what let the old `TemplateProcessingDeserializer` read
    /// an object tagged `"TemplateProcessing"`.
    #[test]
    fn template_tag_is_optional_and_its_value_unchecked() {
        for json in [
            r#"{"single":[{"Sequence":{"id":"A","type_id":0}}],"pair":[{"Sequence":{"id":"A","type_id":0}},{"Sequence":{"id":"B","type_id":1}}],"special_tokens":{}}"#,
            r#"{"type":"whatever","single":[{"Sequence":{"id":"A","type_id":0}}],"pair":[{"Sequence":{"id":"A","type_id":0}},{"Sequence":{"id":"B","type_id":1}}],"special_tokens":{}}"#,
        ] {
            assert!(
                serde_json::from_str::<TemplateProcessingMirror>(json).is_ok(),
                "should have parsed: {json}"
            );
        }

        // A missing required field is still an error, and that is what discriminates the untagged
        // variants from each other.
        match serde_json::from_str::<TemplateProcessingMirror>(
            r#"{"type":"TemplateProcessing","single":[]}"#,
        ) {
            Err(err) => assert_eq!(err.to_string(), "missing field `pair` at line 1 column 41"),
            _ => panic!("a template with no `pair` must not parse"),
        }
    }
}
