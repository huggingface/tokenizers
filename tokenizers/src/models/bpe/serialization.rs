use super::{
    super::OrderedVocabIter, convert_merges_to_hashmap, parse_legacy_merge, resolve_merge,
    BpeBuilder, MergeInput, MergeMap, Merges, Pair, BPE,
};
use ahash::AHashMap;
use serde::{
    de::{Error, MapAccess, Visitor},
    ser::SerializeStruct,
    Deserialize, Deserializer, Serialize, Serializer,
};
use std::borrow::Cow;

impl Serialize for BPE {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut model = serializer.serialize_struct("BPE", 8)?;

        // Start by small fields
        model.serialize_field("type", "BPE")?;
        model.serialize_field("dropout", &self.dropout)?;
        model.serialize_field("unk_token", &self.unk_token)?;
        model.serialize_field("continuing_subword_prefix", &self.continuing_subword_prefix)?;
        model.serialize_field("end_of_word_suffix", &self.end_of_word_suffix)?;
        model.serialize_field("fuse_unk", &self.fuse_unk)?;
        model.serialize_field("byte_fallback", &self.byte_fallback)?;
        model.serialize_field("ignore_merges", &self.ignore_merges)?;

        // Then the large ones
        let mut merges: Vec<(&Pair, &u32)> = self
            .merges
            .iter()
            .map(|(pair, (rank, _))| (pair, rank))
            .collect();
        merges.sort_unstable_by_key(|k| *k.1);
        let merges = merges
            .into_iter()
            .map(|(pair, _)| (self.vocab_r[&pair.0].clone(), self.vocab_r[&pair.1].clone()))
            .collect::<Vec<_>>();
        let ordered_vocab = OrderedVocabIter::new(&self.vocab_r);

        model.serialize_field("vocab", &ordered_vocab)?;
        model.serialize_field("merges", &merges)?;

        model.end()
    }
}

impl<'de> Deserialize<'de> for BPE {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_struct(
            "BPE",
            &[
                "type",
                "dropout",
                "unk_token",
                "continuing_subword_prefix",
                "end_of_word_suffix",
                "fuse_unk",
                "byte_fallback",
                "ignore_merges",
                "vocab",
                "merges",
            ],
            BPEVisitor,
        )
    }
}

struct BPEVisitor;
impl<'de> Visitor<'de> for BPEVisitor {
    type Value = BPE;

    fn expecting(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(fmt, "struct BPE")
    }

    fn visit_map<V>(self, mut map: V) -> std::result::Result<Self::Value, V::Error>
    where
        V: MapAccess<'de>,
    {
        let mut builder = BpeBuilder::new();
        let mut vocab: Option<AHashMap<String, u32>> = None;
        // Needed to resolve merges on the fly; `None` until the field is seen.
        let mut prefix_len: Option<usize> = None;
        let mut merges: Option<MergeInput> = None;

        while let Some(key) = map.next_key::<String>()? {
            match key.as_ref() {
                "dropout" => {
                    if let Some(dropout) = map.next_value()? {
                        builder = builder.dropout(dropout);
                    }
                }
                "unk_token" => {
                    if let Some(unk) = map.next_value()? {
                        builder = builder.unk_token(unk);
                    }
                }
                "continuing_subword_prefix" => {
                    let prefix: Option<String> = map.next_value()?;
                    prefix_len = Some(prefix.as_ref().map_or(0, String::len));
                    if let Some(prefix) = prefix {
                        builder = builder.continuing_subword_prefix(prefix);
                    }
                }
                "end_of_word_suffix" => {
                    if let Some(suffix) = map.next_value()? {
                        builder = builder.end_of_word_suffix(suffix);
                    }
                }
                "fuse_unk" => {
                    if let Some(suffix) = map.next_value()? {
                        builder = builder.fuse_unk(suffix);
                    }
                }
                "byte_fallback" => {
                    if let Some(suffix) = map.next_value()? {
                        builder = builder.byte_fallback(suffix);
                    }
                }
                "ignore_merges" => {
                    if let Some(suffix) = map.next_value()? {
                        builder = builder.ignore_merges(suffix);
                    }
                }
                "vocab" => vocab = Some(map.next_value()?),
                "merges" => {
                    merges = Some(match (&vocab, prefix_len) {
                        // Fast path: resolve merges to ids as they are parsed.
                        (Some(vocab), Some(prefix_len)) => {
                            let max_len = vocab.keys().map(|k| k.len()).max().unwrap_or(0);
                            MergeInput::Resolved(map.next_value_seed(MergesResolver {
                                vocab,
                                prefix_len,
                                max_len,
                            })?)
                        }
                        // vocab/prefix not seen yet: buffer raw, resolve in `build`.
                        _ => MergeInput::Raw(map.next_value::<RawMergeType>()?.into_pairs()?),
                    });
                }
                "type" => match map.next_value()? {
                    "BPE" => {}
                    u => return Err(Error::invalid_value(serde::de::Unexpected::Str(u), &"BPE")),
                },
                _ => {}
            }
        }
        match (vocab, merges) {
            (Some(vocab), Some(merges)) => builder
                .vocab_and_merge_input(vocab, merges)
                .build()
                .map_err(Error::custom),
            _ => Err(Error::custom("Missing vocab/merges")),
        }
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum RawMergeType {
    Tuple(Vec<(String, String)>),
    Legacy(Vec<String>),
}

impl RawMergeType {
    fn into_pairs<E: Error>(self) -> Result<Merges, E> {
        match self {
            RawMergeType::Tuple(pairs) => Ok(pairs),
            RawMergeType::Legacy(lines) => {
                convert_merges_to_hashmap(lines.into_iter()).map_err(E::custom)
            }
        }
    }
}

/// Deserializes the `merges` array straight into a resolved `MergeMap`, so the
/// merge tokens are never collected into an owned `Vec<(String, String)>`.
struct MergesResolver<'v> {
    vocab: &'v AHashMap<String, u32>,
    prefix_len: usize,
    max_len: usize,
}

impl<'de, 'v> serde::de::DeserializeSeed<'de> for MergesResolver<'v> {
    type Value = MergeMap;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_seq(self)
    }
}

impl<'de, 'v> Visitor<'de> for MergesResolver<'v> {
    type Value = MergeMap;

    fn expecting(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(fmt, "a sequence of BPE merge rules")
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: serde::de::SeqAccess<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum MergeElem<'a> {
            Pair(#[serde(borrow)] (Cow<'a, str>, Cow<'a, str>)),
            Legacy(#[serde(borrow)] Cow<'a, str>),
        }

        let mut buffer: Vec<u8> = vec![0; self.max_len];
        let mut merge_map =
            MergeMap::with_capacity_and_hasher(seq.size_hint().unwrap_or(0), Default::default());
        let mut rank: u32 = 0;
        while let Some(elem) = seq.next_element::<MergeElem>()? {
            let (a, b): (Cow<str>, Cow<str>) = match elem {
                MergeElem::Pair((a, b)) => (a, b),
                MergeElem::Legacy(line) => {
                    match parse_legacy_merge(&line, rank as usize + 1).map_err(Error::custom)? {
                        Some((a, b)) => (Cow::Owned(a), Cow::Owned(b)),
                        None => continue,
                    }
                }
            };
            let (pair, value) =
                resolve_merge(self.vocab, &mut buffer, self.prefix_len, rank, &a, &b)
                    .map_err(Error::custom)?;
            merge_map.insert(pair, value);
            rank += 1;
        }
        Ok(merge_map)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::models::bpe::Vocab;

    #[test]
    fn test_serialization() {
        let vocab: Vocab = [
            ("<unk>".into(), 0),
            ("a".into(), 1),
            ("b".into(), 2),
            ("ab".into(), 3),
        ]
        .iter()
        .cloned()
        .collect();
        let bpe = BpeBuilder::default()
            .vocab_and_merges(vocab, vec![("a".to_string(), "b".to_string())])
            .unk_token("<unk>".to_string())
            .ignore_merges(true)
            .build()
            .unwrap();

        let legacy = r#"{"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":true,"vocab":{"<unk>":0,"a":1,"b":2,"ab":3},"merges":["a b"]}"#;
        let legacy = serde_json::from_str(legacy).unwrap();
        assert_eq!(bpe, legacy);

        let data = serde_json::to_string(&bpe).unwrap();
        assert_eq!(
            data,
            r#"{"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":true,"vocab":{"<unk>":0,"a":1,"b":2,"ab":3},"merges":[["a","b"]]}"#
        );
        let reconstructed = serde_json::from_str(&data).unwrap();
        assert_eq!(bpe, reconstructed);

        // With a space in the token
        let vocab: Vocab = [
            ("<unk>".into(), 0),
            ("a".into(), 1),
            ("b c d".into(), 2),
            ("ab c d".into(), 3),
        ]
        .iter()
        .cloned()
        .collect();
        let bpe = BpeBuilder::default()
            .vocab_and_merges(vocab, vec![("a".to_string(), "b c d".to_string())])
            .unk_token("<unk>".to_string())
            .ignore_merges(true)
            .build()
            .unwrap();
        let data = serde_json::to_string(&bpe).unwrap();
        assert_eq!(
            data,
            r#"{"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":true,"vocab":{"<unk>":0,"a":1,"b c d":2,"ab c d":3},"merges":[["a","b c d"]]}"#
        );
        let reconstructed = serde_json::from_str(&data).unwrap();
        assert_eq!(bpe, reconstructed);
    }

    #[test]
    fn test_serialization_ignore_merges() {
        let vocab: Vocab = [("<unk>".into(), 0), ("a".into(), 1), ("b".into(), 2)]
            .iter()
            .cloned()
            .collect();
        let mut bpe = BpeBuilder::default()
            .vocab_and_merges(vocab, vec![])
            .unk_token("<unk>".to_string())
            .ignore_merges(true)
            .build()
            .unwrap();

        let bpe_string = r#"{"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":true,"vocab":{"<unk>":0,"a":1,"b":2},"merges":[]}"#;
        assert_eq!(serde_json::from_str::<BPE>(bpe_string).unwrap(), bpe);

        bpe.ignore_merges = false;
        let bpe_string = r#"{"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"vocab":{"<unk>":0,"a":1,"b":2},"merges":[]}"#;
        assert_eq!(serde_json::from_str::<BPE>(bpe_string).unwrap(), bpe);
    }

    // Deserialization must work independent of JSON field order.
    #[test]
    fn test_deserialize_field_order_independent() {
        use itertools::Itertools;

        let fields = [
            r#""type":"BPE""#,
            r#""dropout":null"#,
            r#""unk_token":"[UNK]""#,
            r###""continuing_subword_prefix":"##""###,
            r#""end_of_word_suffix":null"#,
            r#""fuse_unk":false"#,
            r#""byte_fallback":false"#,
            r#""ignore_merges":true"#,
            r###""vocab":{"[UNK]":0,"a":1,"##b":2,"##c":3,"ab":4,"abc":5}"###,
            r###""merges":[["a","##b"],["ab","##c"]]"###,
        ];

        let expected: BPE =
            serde_json::from_str(&format!("{{{}}}", fields.iter().join(","))).unwrap();

        for i in 0..fields.len() {
            let mut rotated = fields.iter().cycle().skip(i).take(fields.len());
            let json = format!("{{{}}}", rotated.join(","));
            let bpe: BPE = serde_json::from_str(&json)
                .unwrap_or_else(|e| panic!("failed to deserialize {}: {}", json, e));
            assert_eq!(bpe, expected, "field order {} changed the model", json);
        }
    }

    // Legacy "a b" merges must parse the same whether merges come before or
    // after vocab (fallback vs fast path).
    #[test]
    fn test_deserialize_legacy_merges_both_orders() {
        let vocab = r#""vocab":{"a":0,"b":1,"ab":2}"#;
        let merges = r#""merges":["a b"]"#;
        let vocab_first: BPE =
            serde_json::from_str(&format!(r#"{{"type":"BPE",{vocab},{merges}}}"#)).unwrap();
        let merges_first: BPE =
            serde_json::from_str(&format!(r#"{{"type":"BPE",{merges},{vocab}}}"#)).unwrap();
        assert_eq!(vocab_first, merges_first);
    }

    // A merge referencing a token outside the vocab is rejected.
    #[test]
    fn test_deserialize_merge_out_of_vocab() {
        let json = r#"{"type":"BPE","vocab":{"a":0,"b":1,"ab":2},"merges":[["a","zzz"]]}"#;
        let err = serde_json::from_str::<BPE>(json).unwrap_err();
        assert!(
            err.to_string().starts_with("Token `zzz` out of vocabulary"),
            "unexpected error: {}",
            err
        );
    }
}
