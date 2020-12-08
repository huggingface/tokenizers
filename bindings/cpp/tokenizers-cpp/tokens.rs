#[cxx::bridge(namespace = "huggingface::tokenizers")]
pub mod ffi {
    #[derive(PartialEq, Eq, Hash, Default, Debug, Clone)]
    pub struct Token {
        pub id: u32,
        pub value: String,
        pub start: usize,
        pub end: usize,
    }

    #[derive(PartialEq, Eq, Hash, Default, Debug, Clone)]
    pub struct Tokens {
        pub tokens: Vec<Token>,
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

impl From<Vec<tk::Token>> for Tokens {
    fn from(tokens: Vec<tk::Token>) -> Self {
        Tokens {
            tokens: tokens.into_iter().map(|token| token.into()).collect(),
        }
    }
}

impl From<&Vec<tk::Token>> for Tokens {
    fn from(tokens: &Vec<tk::Token>) -> Self {
        Tokens {
            tokens: tokens.into_iter().map(|token| token.into()).collect(),
        }
    }
}
