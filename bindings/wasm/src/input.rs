use serde::Deserialize;
use tokenizers::pipeline::{Input, Inputs};
use tsify::Tsify;

/// A text, or a pair of texts.
#[derive(Tsify, Deserialize)]
#[serde(untagged)]
pub enum TextInput {
    Single(String),
    Pair([String; 2]),
}

impl From<TextInput> for Input {
    fn from(input: TextInput) -> Self {
        match input {
            TextInput::Single(text) => Input::Single(text),
            TextInput::Pair([first, second]) => Input::Pair(first, second),
        }
    }
}

impl From<TextInput> for Inputs {
    fn from(input: TextInput) -> Self {
        Self::Single(input.into())
    }
}
