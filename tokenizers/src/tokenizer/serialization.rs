use std::marker::PhantomData;

use serde::{
    self,
    de::{Error, MapAccess, Visitor},
    ser::SerializeStruct,
    Deserialize, Deserializer, Serialize, Serializer,
};

use super::{added_vocabulary::AddedTokenWithId, TokenizerImpl};
use crate::{Decoder, Model, Normalizer, PostProcessor, PreTokenizer, TokenizerBuilder};

static SERIALIZATION_VERSION: &str = "1.0";

impl<M, N, PT, PP, D> Serialize for TokenizerImpl<M, N, PT, PP, D>
where
    M: Serialize,
    N: Serialize,
    PT: Serialize,
    PP: Serialize,
    D: Serialize,
{
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
        tokenizer.serialize_field("added_tokens", &self.added_vocabulary)?;

        // Then add our parts
        tokenizer.serialize_field("normalizer", &self.normalizer)?;
        tokenizer.serialize_field("pre_tokenizer", &self.pre_tokenizer)?;
        tokenizer.serialize_field("post_processor", &self.post_processor)?;
        tokenizer.serialize_field("decoder", &self.decoder)?;
        tokenizer.serialize_field("model", &self.model)?;

        tokenizer.end()
    }
}

impl<'de, M, N, PT, PP, D> Deserialize<'de> for TokenizerImpl<M, N, PT, PP, D>
where
    M: Deserialize<'de> + Model,
    N: Deserialize<'de> + Normalizer,
    PT: Deserialize<'de> + PreTokenizer,
    PP: Deserialize<'de> + PostProcessor,
    D: Deserialize<'de> + Decoder,
{
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: Deserializer<'de>,
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
            TokenizerVisitor(
                PhantomData,
                PhantomData,
                PhantomData,
                PhantomData,
                PhantomData,
            ),
        )
    }
}

struct TokenizerVisitor<M, N, PT, PP, D>(
    PhantomData<M>,
    PhantomData<N>,
    PhantomData<PT>,
    PhantomData<PP>,
    PhantomData<D>,
);

impl<'de, M, N, PT, PP, D> Visitor<'de> for TokenizerVisitor<M, N, PT, PP, D>
where
    M: Deserialize<'de> + Model,
    N: Deserialize<'de> + Normalizer,
    PT: Deserialize<'de> + PreTokenizer,
    PP: Deserialize<'de> + PostProcessor,
    D: Deserialize<'de> + Decoder,
{
    type Value = TokenizerImpl<M, N, PT, PP, D>;

    fn expecting(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(fmt, "struct Tokenizer")
    }

    fn visit_map<V>(self, mut map: V) -> Result<Self::Value, V::Error>
    where
        V: MapAccess<'de>,
    {
        let mut builder = TokenizerBuilder::new();
        let mut tokens: Vec<AddedTokenWithId> = vec![];
        while let Some(key) = map.next_key::<String>()? {
            match key.as_ref() {
                "version" => {
                    let v: String = map.next_value()?;
                    if &v != "1.0" {
                        return Err(Error::custom(format!("Unknown tokenizer version '{}'", v)));
                    }
                }
                "truncation" => {
                    builder = builder.with_truncation(map.next_value()?);
                }
                "padding" => {
                    builder = builder.with_padding(map.next_value()?);
                }
                "added_tokens" => {
                    tokens = map.next_value()?;
                }
                "normalizer" => {
                    builder = builder.with_normalizer(map.next_value()?);
                }
                "pre_tokenizer" => {
                    builder = builder.with_pre_tokenizer(map.next_value()?);
                }
                "model" => {
                    builder = builder.with_model(map.next_value()?);
                }
                "decoder" => {
                    builder = builder.with_decoder(map.next_value()?);
                }
                "post_processor" => {
                    builder = builder.with_post_processor(map.next_value()?);
                }
                _ => {}
            };
        }
        let mut tokenizer = builder
            .build()
            .map_err(|e| V::Error::custom(e.to_string()))?;

        // We take care of deserializing the added_tokens (instead of `AddedVocabulary` directly
        // because it let us check that associated IDs are still good, and warn the user otherwise
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
                warn!(
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
