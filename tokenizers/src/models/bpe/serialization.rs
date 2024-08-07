use super::{super::OrderedVocabIter, convert_merges_to_hashmap, BpeBuilder, Pair, BPE};
use serde::{
    de::{Error, MapAccess, Visitor},
    ser::SerializeStruct,
    Deserialize, Deserializer, Serialize, Serializer,
};
use std::collections::HashMap;

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
        let mut vocab: Option<HashMap<String, u32>> = None;

        #[derive(Debug, Deserialize)]
        #[serde(untagged)]
        enum MergeType {
            Tuple(Vec<(String, String)>),
            Legacy(Vec<String>),
        }
        let mut merges: Option<MergeType> = None;
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
                    if let Some(prefix) = map.next_value()? {
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
                "merges" => merges = Some(map.next_value()?),
                "type" => match map.next_value()? {
                    "BPE" => {}
                    u => {
                        return Err(serde::de::Error::invalid_value(
                            serde::de::Unexpected::Str(u),
                            &"BPE",
                        ))
                    }
                },
                _ => {}
            }
        }
        if let (Some(vocab), Some(merges)) = (vocab, merges) {
            let merges = match merges {
                MergeType::Tuple(merges) => merges,
                MergeType::Legacy(merges) => {
                    convert_merges_to_hashmap(merges.into_iter(), &vocab).map_err(Error::custom)?
                }
            };
            builder = builder.vocab_and_merges(vocab, merges);
            Ok(builder.build().map_err(Error::custom)?)
        } else {
            Err(Error::custom("Missing vocab/merges"))
        }
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
}
