#![warn(clippy::all)]

extern crate neon;
extern crate neon_serde;
extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate tokenizers as tk;

mod container;
mod decoders;
mod encoding;
mod models;
mod normalizers;
mod pre_tokenizers;
mod processors;
mod tasks;
mod tokenizer;
mod trainers;
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
    // PreTokenizers
    pre_tokenizers::register(&mut m, "pre_tokenizers")?;
    // Trainers
    trainers::register(&mut m, "trainers")?;
    // Utils
    utils::register(&mut m, "utils")?;

    Ok(())
});
