extern crate neon;
extern crate tokenizers as tk;

mod decoders;
mod encoding;
mod models;
mod normalizers;
mod processors;
mod tasks;
mod tokenizer;
mod utils;

use neon::prelude::*;

register_module!(mut m, {
    // Tokenizer
    tokenizer::register(&mut m, "tokenizer")?;
    // Models
    models::register(&mut m, "models")?;
    // Decoders
    decoders::register(&mut m, "decoders")?;
    // Processors
    processors::register(&mut m, "processors")?;
    // Normalizers
    normalizers::register(&mut m, "normalizers")?;

    Ok(())
});
