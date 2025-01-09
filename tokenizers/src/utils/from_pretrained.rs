use crate::Result;
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
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
            buf.push(format!("'{}'", x));
            buf
        })
        .join(", "); // "'/', '-', '_', '.'"
    if !valid {
        return Err(format!(
            "Model \"{}\" contains invalid characters, expected only alphanumeric or {valid_chars_stringified}",
            identifier
        )
        .into());
    }
    let params = params.unwrap_or_default();

    let revision = &params.revision;
    let valid_revision = revision.chars().all(is_valid_char);
    if !valid_revision {
        return Err(format!(
            "Revision \"{}\" contains invalid characters, expected only alphanumeric or {valid_chars_stringified}",
            revision
        )
        .into());
    }

    let mut builder = ApiBuilder::new();
    if let Some(token) = params.token {
        builder = builder.with_token(Some(token));
    }
    let api = builder.build()?;
    let repo = Repo::with_revision(identifier, RepoType::Model, params.revision);
    let api = api.repo(repo);
    Ok(api.get("tokenizer.json")?)
}
