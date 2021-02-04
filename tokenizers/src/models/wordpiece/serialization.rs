use super::{super::OrderedVocabIter, WordPiece, WordPieceBuilder};
use serde::{
    de::{MapAccess, Visitor},
    ser::SerializeStruct,
    Deserialize, Deserializer, Serialize, Serializer,
};
use std::collections::HashSet;

impl Serialize for WordPiece {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut model = serializer.serialize_struct("WordPiece", 5)?;

        // Small fields first
        model.serialize_field("type", "WordPiece")?;
        model.serialize_field("unk_token", &self.unk_token)?;
        model.serialize_field("continuing_subword_prefix", &self.continuing_subword_prefix)?;
        model.serialize_field("max_input_chars_per_word", &self.max_input_chars_per_word)?;

        // Then large ones
        let ordered_vocab = OrderedVocabIter::new(&self.vocab_r);
        model.serialize_field("vocab", &ordered_vocab)?;

        model.end()
    }
}

impl<'de> Deserialize<'de> for WordPiece {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_struct(
            "WordPiece",
            &[
                "type",
                "unk_token",
                "continuing_subword_prefix",
                "max_input_chars_per_word",
                "vocab",
            ],
            WordPieceVisitor,
        )
    }
}

struct WordPieceVisitor;
impl<'de> Visitor<'de> for WordPieceVisitor {
    type Value = WordPiece;

    fn expecting(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(fmt, "struct WordPiece")
    }

    fn visit_map<V>(self, mut map: V) -> std::result::Result<Self::Value, V::Error>
    where
        V: MapAccess<'de>,
    {
        let mut builder = WordPieceBuilder::new();
        let mut missing_fields = vec![
            // for retrocompatibility the "type" field is not mandatory
            "unk_token",
            "continuing_subword_prefix",
            "max_input_chars_per_word",
            "vocab",
        ]
        .into_iter()
        .collect::<HashSet<_>>();

        while let Some(key) = map.next_key::<String>()? {
            match key.as_ref() {
                "unk_token" => builder = builder.unk_token(map.next_value()?),
                "continuing_subword_prefix" => {
                    builder = builder.continuing_subword_prefix(map.next_value()?)
                }
                "max_input_chars_per_word" => {
                    builder = builder.max_input_chars_per_word(map.next_value()?)
                }
                "vocab" => builder = builder.vocab(map.next_value()?),
                "type" => match map.next_value()? {
                    "WordPiece" => {}
                    u => {
                        return Err(serde::de::Error::invalid_value(
                            serde::de::Unexpected::Str(u),
                            &"WordPiece",
                        ))
                    }
                },
                _ => {}
            }
            missing_fields.remove::<str>(&key);
        }

        if !missing_fields.is_empty() {
            Err(serde::de::Error::missing_field(
                missing_fields.iter().next().unwrap(),
            ))
        } else {
            Ok(builder.build().map_err(serde::de::Error::custom)?)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serde() {
        let wp = WordPiece::default();
        let wp_s = "{\
            \"type\":\"WordPiece\",\
            \"unk_token\":\"[UNK]\",\
            \"continuing_subword_prefix\":\"##\",\
            \"max_input_chars_per_word\":100,\
            \"vocab\":{}\
        }";

        assert_eq!(serde_json::to_string(&wp).unwrap(), wp_s);
        assert_eq!(serde_json::from_str::<WordPiece>(wp_s).unwrap(), wp);
    }

    #[test]
    fn deserialization_should_fail() {
        let missing_unk = "{\
            \"type\":\"WordPiece\",\
            \"continuing_subword_prefix\":\"##\",\
            \"max_input_chars_per_word\":100,\
            \"vocab\":{}\
        }";
        assert!(serde_json::from_str::<WordPiece>(missing_unk)
            .unwrap_err()
            .to_string()
            .starts_with("missing field `unk_token`"));

        let wrong_type = "{\
            \"type\":\"WordLevel\",\
            \"unk_token\":\"[UNK]\",\
            \"vocab\":{}\
        }";
        assert!(serde_json::from_str::<WordPiece>(wrong_type)
            .unwrap_err()
            .to_string()
            .starts_with("invalid value: string \"WordLevel\", expected WordPiece"));
    }
}
