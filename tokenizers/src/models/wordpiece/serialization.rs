use super::{super::OrderedVocabIter, WordPiece};
use serde::{ser::SerializeStruct, Deserialize, Deserializer, Serialize, Serializer};

impl Serialize for WordPiece {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut model = serializer.serialize_struct("WordPiece", 4)?;

        // Small fields first
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
        unimplemented!()
    }
}
