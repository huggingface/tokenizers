extern crate tokenizers as tk;

use crate::utils::Container;
use neon::prelude::*;

/// Encoding
pub struct Encoding {
    pub encoding: Container<tk::tokenizer::Encoding>,
}

declare_types! {
    pub class JsEncoding for Encoding {
        init(_) {
            // This should never be called from JavaScript
            Ok(Encoding {
                encoding: Container::Empty
            })
        }
    }
}
