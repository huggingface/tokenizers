use std::marker::PhantomData;

use serde::{
    self,
    de::{Error, IgnoredAny, MapAccess, Visitor},
    ser::SerializeStruct,
    Deserialize, Deserializer, Serialize, Serializer,
};

use super::added_vocabulary::AddedTokenWithId;
use super::TokenizerImpl;
use crate::{Model, Normalizer, PreTokenizer, TokenizerBuilder};

static SERIALIZATION_VERSION: &str = "1.0";

impl<M, N, PT> Serialize for TokenizerImpl<M, N, PT>
where
    M: Serialize,
    N: Serialize,
    PT: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut tokenizer = serializer.serialize_struct("Tokenizer", 5)?;

        // Start by adding the current version
        tokenizer.serialize_field("version", SERIALIZATION_VERSION)?;

        // Added tokens
        tokenizer.serialize_field("added_tokens", &self.added_vocabulary)?;

        // Then add our parts. The pipeline-only tokenizer does not carry truncation,
        // padding, a post-processor or a decoder, so those fields are not emitted.
        tokenizer.serialize_field("normalizer", &self.normalizer)?;
        tokenizer.serialize_field("pre_tokenizer", &self.pre_tokenizer)?;
        tokenizer.serialize_field("model", &self.model)?;

        tokenizer.end()
    }
}

impl<'de, M, N, PT> Deserialize<'de> for TokenizerImpl<M, N, PT>
where
    M: Deserialize<'de> + Model,
    N: Deserialize<'de> + Normalizer,
    PT: Deserialize<'de> + PreTokenizer,
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
            TokenizerVisitor(PhantomData, PhantomData, PhantomData),
        )
    }
}

struct TokenizerVisitor<M, N, PT>(PhantomData<M>, PhantomData<N>, PhantomData<PT>);

impl<'de, M, N, PT> Visitor<'de> for TokenizerVisitor<M, N, PT>
where
    M: Deserialize<'de> + Model,
    N: Deserialize<'de> + Normalizer,
    PT: Deserialize<'de> + PreTokenizer,
{
    type Value = TokenizerImpl<M, N, PT>;

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
                        return Err(Error::custom(format!("Unknown tokenizer version '{v}'")));
                    }
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
                // `truncation`, `padding`, `post_processor` and `decoder` are part of the
                // tokenizer.json schema but are not used by the pipeline-only tokenizer;
                // consume and discard them so loading a full tokenizer.json still works.
                _ => {
                    let _ = map.next_value::<IgnoredAny>()?;
                }
            };
        }
        let mut tokenizer = builder
            .build()
            .map_err(|e| V::Error::custom(e.to_string()))?;

        // Single-pass: warn on ID mismatches, then add all tokens.
        // `add_tokens` computes normalization internally for tokens with `normalized = true`.
        for t in &tokens {
            if let Some(rid) = tokenizer.token_to_id(&t.token.content) {
                if rid != t.id {
                    warn!(
                        "Warning: Token '{}' was expected to have ID '{}' but was given ID '{}'",
                        t.token.content, t.id, rid
                    );
                }
            }
        }
        tokenizer
            .add_tokens(tokens.into_iter().map(|t| t.token))
            .map_err(|e| V::Error::custom(e.to_string()))?;

        Ok(tokenizer)
    }
}

#[cfg(test)]
mod tests {
    use crate::tokenizer::Tokenizer;
    use std::str::FromStr;

    #[test]
    fn test_deserialization_serialization_invariant() {
        // The pipeline-only tokenizer serializes exactly these fields; `truncation`,
        // `padding`, `post_processor` and `decoder` are ignored on load and not emitted.
        let tok_json = r#"{
  "version": "1.0",
  "added_tokens": [
    {
      "id": 0,
      "content": "[SPECIAL_0]",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    },
    {
      "id": 1,
      "content": "[SPECIAL_1]",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": true,
      "special": false
    },
    {
      "id": 2,
      "content": "[SPECIAL_2]",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    }
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "",
    "max_input_chars_per_word": 100,
    "vocab": {}
  }
}"#;
        let tokenizer = Tokenizer::from_str(tok_json).unwrap();

        let tok_str = serde_json::to_string_pretty(&tokenizer).unwrap();
        // It should be exactly the same as above
        assert_eq!(tok_str, tok_json);
    }

    #[cfg(feature = "http")]
    #[test]
    fn test_from_pretrained() {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .with_target(false)
            .init();
        let _ = Tokenizer::from_pretrained("Qwen/Qwen2-7B-Instruct", None);
        warn!("This should be the first warning");
    }
}
