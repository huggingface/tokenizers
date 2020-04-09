use super::{super::OrderedVocabIter, WordLevel};
use serde::{ser::SerializeStruct, Deserialize, Deserializer, Serialize, Serializer};

impl Serialize for WordLevel {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut model = serializer.serialize_struct("WordLevel", 2)?;
        let ordered_vocab = OrderedVocabIter::new(&self.vocab_r);
        model.serialize_field("vocab", &ordered_vocab)?;
        model.serialize_field("unk_token", &self.unk_token)?;
        model.end()
    }
}

impl<'de> Deserialize<'de> for WordLevel {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        unimplemented!()
    }
}
