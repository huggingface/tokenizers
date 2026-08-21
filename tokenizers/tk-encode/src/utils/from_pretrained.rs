use crate::Result;
use hf_hub::{HFClientBuilder, split_id};
use std::collections::HashMap;
use std::path::PathBuf;

/// Defines the additional parameters available for the `from_pretrained` function
#[derive(Debug, Clone)]
pub struct FromPretrainedParameters {
    pub revision: String,
    pub user_agent: HashMap<String, String>,
    pub token: Option<String>,
}

impl Default for FromPretrainedParameters {
    fn default() -> Self {
        Self {
            revision: "main".into(),
            user_agent: HashMap::new(),
            token: None,
        }
    }
}

/// Downloads and cache the identified tokenizer if it exists on
/// the Hugging Face Hub, and returns a local path to the file
pub fn from_pretrained<S: AsRef<str>>(
    identifier: S,
    params: Option<FromPretrainedParameters>,
) -> Result<PathBuf> {
    let identifier: String = identifier.as_ref().to_string();

    let valid_chars = ['-', '_', '.', '/'];
    let is_valid_char = |x: char| x.is_alphanumeric() || valid_chars.contains(&x);

    let valid = identifier.chars().all(is_valid_char);
    let valid_chars_stringified = valid_chars
        .iter()
        .fold(vec![], |mut buf, x| {
            buf.push(format!("'{x}'"));
            buf
        })
        .join(", "); // "'/', '-', '_', '.'"
    if !valid {
        return Err(format!(
            "Model \"{identifier}\" contains invalid characters, expected only alphanumeric or {valid_chars_stringified}"
        )
        .into());
    }
    let params = params.unwrap_or_default();

    let revision = &params.revision;
    let valid_revision = revision.chars().all(is_valid_char);
    if !valid_revision {
        return Err(format!(
            "Revision \"{revision}\" contains invalid characters, expected only alphanumeric or {valid_chars_stringified}"
        )
        .into());
    }

    // `HFClientBuilder::new()` is the `ApiBuilder::from_env()` of hf-hub 1.x: endpoint, cache dir
    // and token all fall back to the environment (`HF_ENDPOINT`, `HF_HOME`, `HF_TOKEN`) and to the
    // token file `huggingface-cli login` writes, and only at `build` time.
    let mut builder = HFClientBuilder::new();
    if let Some(token) = params.token {
        builder = builder.token(token);
    }
    let client = builder.build_sync()?;

    // hf-hub 1.x addresses a repo as owner + name rather than one string. `split_id` yields an
    // empty owner for a short-form id like `gpt2`, which the repo handle accepts as-is.
    let (owner, name) = split_id(&identifier);
    Ok(client
        .model(owner, name)
        .download_file()
        .filename("tokenizer.json")
        .revision(params.revision)
        .send()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Salvaged from the umbrella crate's `tests/from_pretrained.rs`, which `autotests = false`
    /// meant was never compiled. Neither case reaches the network: both are refused by the
    /// character checks above, before a client is ever built.
    #[test]
    fn invalid_characters_are_refused_before_any_request() {
        assert!(from_pretrained("docs?", None).is_err());
        assert!(
            from_pretrained(
                "bert-base-cased",
                Some(FromPretrainedParameters {
                    revision: "gpt?".to_string(),
                    ..Default::default()
                }),
            )
            .is_err()
        );
    }
}
