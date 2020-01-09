extern crate neon;
extern crate tokenizers as tk;

mod models;
mod tokenizer;
mod utils;

use neon::prelude::*;

register_module!(mut m, {
    // Tokenizer
    tokenizer::register(&mut m, "tokenizer")?;
    // Models
    models::register(&mut m, "models")?;

    Ok(())
});
