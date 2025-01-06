use super::{
    super::bpe::Pair, super::OrderedVocabIter, convert_merges_to_hashmap, BacktrackingBpe,
    BacktrackingBpeBuilder,
};
use regex_syntax::ast::print;
use serde::{
    de::{Error, MapAccess, Visitor},
    ser::SerializeStruct,
    Deserialize, Deserializer, Serialize, Serializer,
};
use std::collections::HashMap;

impl Serialize for BacktrackingBpe {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut model = serializer.serialize_struct("BPE", 8)?;

        // Start by small fields
        model.serialize_field("type", "BPE")?;

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

impl<'de> Deserialize<'de> for BacktrackingBpe {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_struct(
            "BPE",
            &["type", "dropout", "unk_token", "vocab", "merges"],
            BacktrackingBpeVisitor,
        )
    }
}

struct BacktrackingBpeVisitor;
impl<'de> Visitor<'de> for BacktrackingBpeVisitor {
    type Value = BacktrackingBpe;

    fn expecting(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(fmt, "struct BacktrackingBpe to be the type")
    }

    fn visit_map<V>(self, mut map: V) -> std::result::Result<Self::Value, V::Error>
    where
        V: MapAccess<'de>,
    {
        let mut builder = BacktrackingBpeBuilder::new();
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
                "vocab" => vocab =  Some(map.next_value()?),
                "merges" => merges = Some(map.next_value()?),
                "type" => match map.next_value()? {
                    "BacktrackingBpe" => {}
                    "BPE" => {println!("Type is BPE but initializing a backtracking BPE")}
                    u => {
                        return Err(serde::de::Error::invalid_value(
                            serde::de::Unexpected::Str(u),
                            &"BacktrackingBpe should have been found",
                        ))
                    }
                },
                field => {
                    println!("Ignoring unused field {:?}", field); // TODO make it into a logger
                    // Ensure the value is consumed to maintain valid deserialization
                    let _ = map.next_value::<serde::de::IgnoredAny>()?;
                }
            }
        }
        if let (Some(vocab), Some(merges)) = (vocab, merges) {
            let merges = match merges {
                MergeType::Tuple(merges) => merges,
                MergeType::Legacy(merges) => {
                    convert_merges_to_hashmap(merges.into_iter(), &vocab).map_err(|e| Error::custom("Error in convert merges to hashmap"))?
                }
            };
            builder = builder.vocab_and_merges(vocab, merges);
            Ok(builder.build().map_err(|e| Error::custom(format!("Error building the backtraciing BPE {:?}", e)))?)
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
        let bpe_string = r#"{
            "type": "BPE",
            "dropout": null,
            "unk_token": "<unk>",
            "continuing_subword_prefix": null,
            "end_of_word_suffix": null,
            "fuse_unk": false,
            "byte_fallback": false,
            "ignore_merges": true,
            "vocab": {
                "a": 1,
                "b c d": 2,
                "ab c d": 3
            },
            "merges": [
                ["a", "b c d"]
            ]
        }"#;
        let reconstructed: Result<BacktrackingBpe, serde_json::Error> = serde_json::from_str(&bpe_string);
        println!("End of my example");



        let vocab: Vocab = [
            ("a".into(), 1),
            ("b".into(), 2),
            ("ab".into(), 3),
        ]
        .iter()
        .cloned()
        .collect();
        let bpe = BacktrackingBpeBuilder::default()
            .vocab_and_merges(vocab, vec![("a".to_string(), "b".to_string())])
            .unk_token("<unk>".to_string())
            .build()
            .unwrap();

        match reconstructed {
            Ok(reconstructed) => {
                println!("Good");
                assert_eq!(bpe, reconstructed);
            }
            Err(err) => {
                println!("Error deserializing: {:?}", err);

            }
        }
    
        let legacy = r#"{"type":"BPE","dropout":null,"unk_token":"<unk>","fuse_unk":false,"byte_fallback":false,"vocab":{"a":1,"b":2,"ab":3},"merges":["a b"]}"#;
        let legacy = serde_json::from_str(legacy);
        match legacy {
            Ok(_) => {
                println!("Good");
                assert_eq!(bpe, legacy.unwrap());
            }
            Err(err) => {
                println!("Error: {:?}", err);
            }
        }
        

        let data = serde_json::to_string(&bpe).unwrap();
        assert_eq!(
            data,
            r#"{"type":"BPE","vocab":{"ab":0,"a":1,"b":2},"merges":[["a","b"]]}"#
        );
        let reconstructed = serde_json::from_str(&data).unwrap();
        assert_eq!(bpe, reconstructed); // TODO failing for now!

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
        let bpe = BacktrackingBpeBuilder::default()
            .vocab_and_merges(vocab, vec![("a".to_string(), "b c d".to_string())])
            .unk_token("<unk>".to_string())
            .build()
            .unwrap();
        let data = serde_json::to_string(&bpe).unwrap();
        assert_eq!(
            data,
            r#"{"type":"BacktrackingBpe","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":true,"vocab":{"<unk>":0,"a":1,"b c d":2,"ab c d":3},"merges":[["a","b c d"]]}"#
        );
        let reconstructed = serde_json::from_str(&data).unwrap();
        assert_eq!(bpe, reconstructed);



    }
}
