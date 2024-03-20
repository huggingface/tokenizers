#[cxx::bridge(namespace = "huggingface::tokenizers")]
pub mod ffi {
    #[derive(PartialEq, Eq, Hash, Default, Debug, Clone)]
    pub struct Token {
        pub id: u32,
        pub value: String,
        pub start: usize,
        pub end: usize,
    }
}

use ffi::*;

impl From<tk::Token> for Token {
    fn from(token: tk::Token) -> Self {
        Token {
            id: token.id,
            value: token.value,
            start: token.offsets.0,
            end: token.offsets.1,
        }
    }
}

impl From<&tk::Token> for Token {
    fn from(token: &tk::Token) -> Self {
        Token {
            id: token.id,
            value: token.value.clone(),
            start: token.offsets.0,
            end: token.offsets.1,
        }
    }
}

pub(crate) fn wrap_tokens(tokens: Vec<tk::Token>) -> Vec<Token> {
    tokens.into_iter().map(|token| token.into()).collect()
}

pub(crate) fn wrap_tokens_ref(tokens: &Vec<tk::Token>) -> Vec<Token> {
    tokens.into_iter().map(|token| token.into()).collect()
}
