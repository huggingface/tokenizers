use super::model::Unigram;
use serde::{
    de::{Error, MapAccess, Visitor},
    ser::SerializeStruct,
    Deserialize, Deserializer, Serialize, Serializer,
};

impl Serialize for Unigram {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut model = serializer.serialize_struct("Unigram", 2)?;

        model.serialize_field("unk_id", &self.unk_id)?;
        model.serialize_field("vocab", &self.vocab)?;

        model.end()
    }
}

impl<'de> Deserialize<'de> for Unigram {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_struct("Unigram", &["vocab", "unk_id"], UnigramVisitor)
    }
}

struct UnigramVisitor;
impl<'de> Visitor<'de> for UnigramVisitor {
    type Value = Unigram;

    fn expecting(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(fmt, "struct Unigram")
    }

    fn visit_map<V>(self, mut map: V) -> std::result::Result<Self::Value, V::Error>
    where
        V: MapAccess<'de>,
    {
        let mut vocab: Option<Vec<(String, f64)>> = None;
        let mut unk_id: Option<usize> = None;
        while let Some(key) = map.next_key::<String>()? {
            match key.as_ref() {
                "unk_id" => {
                    unk_id = map.next_value()?;
                }
                "vocab" => vocab = Some(map.next_value()?),
                _ => (),
            }
        }
        match (vocab, unk_id) {
            (Some(vocab), unk_id) => Ok(Unigram::from(vocab, unk_id)
                .map_err(|err| Error::custom(&format!("Unable to load vocab {:?}", err)))?),
            (None, _) => Err(Error::custom("Missing vocab")),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_serialization() {
        let vocab = vec![("<unk>".to_string(), 0.0), ("a".to_string(), -0.5)];
        let model = Unigram::from(vocab, Some(0)).unwrap();

        let data = serde_json::to_string(&model).unwrap();
        let reconstructed = serde_json::from_str(&data).unwrap();

        assert_eq!(model, reconstructed);
    }

    #[test]
    fn test_serialization_unk_id_not_zero() {
        let vocab = vec![("a".to_string(), -0.5), ("<unk>".to_string(), 0.0)];
        let model = Unigram::from(vocab, Some(1)).unwrap();

        let data = serde_json::to_string(&model).unwrap();
        let reconstructed = serde_json::from_str(&data).unwrap();

        assert_eq!(model, reconstructed);
    }

    #[test]
    fn test_serialization_no_unk_id() {
        let vocab = vec![("a".to_string(), -0.5)];
        let model = Unigram::from(vocab, None).unwrap();

        let data = serde_json::to_string(&model).unwrap();
        let reconstructed = serde_json::from_str(&data).unwrap();

        assert_eq!(model, reconstructed);
    }
}
