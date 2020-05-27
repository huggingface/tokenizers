use super::{AddedToken, Tokenizer};
use crate::models::bpe::BPE;
use serde::{
    self,
    de::{Error, MapAccess, Visitor},
    ser::SerializeStruct,
    Deserialize, Deserializer, Serialize, Serializer,
};

static SERIALIZATION_VERSION: &str = "1.0";

#[derive(Debug, Serialize, Deserialize)]
struct AddedTokenWithId {
    /// The id assigned to this token
    id: u32,
    /// Whether this is a special token
    special: bool,

    #[serde(flatten)]
    /// The target AddedToken
    token: AddedToken,
}

impl Serialize for Tokenizer {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut tokenizer = serializer.serialize_struct("Tokenizer", 9)?;

        // Start by adding the current version
        tokenizer.serialize_field("version", SERIALIZATION_VERSION)?;

        // Params
        tokenizer.serialize_field("truncation", &self.truncation)?;
        tokenizer.serialize_field("padding", &self.padding)?;

        // Added tokens
        let mut added_tokens = self
            .added_tokens_map_r
            .iter()
            .map(|(id, token)| AddedTokenWithId {
                id: *id,
                special: self.special_tokens_set.contains(&token.content),
                token: token.clone(),
            })
            .collect::<Vec<_>>();
        // We need to have these added tokens ordered by ascending ID
        added_tokens.sort_unstable_by_key(|o| o.id);
        tokenizer.serialize_field("added_tokens", &added_tokens)?;

        // Then add our parts
        tokenizer.serialize_field("normalizer", &self.normalizer)?;
        tokenizer.serialize_field("pre_tokenizer", &self.pre_tokenizer)?;
        tokenizer.serialize_field("post_processor", &self.post_processor)?;
        tokenizer.serialize_field("decoder", &self.decoder)?;
        tokenizer.serialize_field("model", &self.model)?;

        tokenizer.end()
    }
}

impl<'de> Deserialize<'de> for Tokenizer {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_struct(
            "Tokenizer",
            &[
                "version",
                "truncation",
                "padding",
                "added_tokens",
                "normalizer",
                "pre_tokenizer",
                "post_processor",
                "decoder",
                "model",
            ],
            TokenizerVisitor,
        )
    }
}

struct TokenizerVisitor;
impl<'de> Visitor<'de> for TokenizerVisitor {
    type Value = Tokenizer;

    fn expecting(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(fmt, "struct Tokenizer")
    }

    fn visit_map<V>(self, mut map: V) -> Result<Self::Value, V::Error>
    where
        V: MapAccess<'de>,
    {
        let mut tokenizer = Tokenizer::new(Box::new(BPE::default()));
        let mut tokens: Vec<AddedTokenWithId> = vec![];
        while let Some(key) = map.next_key()? {
            match key {
                "version" => {
                    let v: &str = map.next_value()?;
                    if v != "1.0" {
                        return Err(Error::custom(format!("Unknown tokenizer version '{}'", v)));
                    }
                }
                "truncation" => {
                    tokenizer.with_truncation(map.next_value()?);
                }
                "padding" => {
                    tokenizer.with_padding(map.next_value()?);
                }
                "added_tokens" => {
                    tokens = map.next_value()?;
                }
                "normalizer" => {
                    if let Some(normalizer) = map.next_value()? {
                        tokenizer.with_normalizer(normalizer);
                    }
                }
                "pre_tokenizer" => {
                    if let Some(pre_tok) = map.next_value()? {
                        tokenizer.with_pre_tokenizer(pre_tok);
                    }
                }
                "model" => {
                    tokenizer.with_model(map.next_value()?);
                }
                "decoder" => {
                    if let Some(decoder) = map.next_value()? {
                        tokenizer.with_decoder(decoder);
                    }
                }
                "post_processor" => {
                    if let Some(processor) = map.next_value()? {
                        tokenizer.with_post_processor(processor);
                    }
                }
                _ => {}
            };
        }

        for token in tokens {
            let tk = token.token.content.clone();
            if token.special {
                tokenizer.add_special_tokens(&[token.token]);
            } else {
                tokenizer.add_tokens(&[token.token]);
            }
            // Warn the user if the id is different than expected
            let received_id = tokenizer.token_to_id(&tk);
            if received_id != Some(token.id) {
                println!(
                    "Warning: Token '{}' was expected to have ID '{}' but was given ID '{}'",
                    tk,
                    token.id,
                    if let Some(rid) = received_id {
                        rid.to_string()
                    } else {
                        "None".to_string()
                    }
                );
            }
        }

        Ok(tokenizer)
    }
}
