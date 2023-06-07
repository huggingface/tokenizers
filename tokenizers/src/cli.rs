//!
//! This is the CLI binary for the Tokenizers project
//!

use clap::{Parser, Subcommand};
use std::io::{self, BufRead, Write};
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::tokenizer::{AddedToken, Result};
use tokenizers::Tokenizer;

/// Generate custom Tokenizers or use existing ones
#[derive(Parser, Debug)]
#[command(author, version)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    Shell {
        /// Path to the vocab.json file
        vocab: String,
        /// Path to the merges.txt file
        merges: String,
    },
}

fn shell(vocab: &str, merges: &str) -> Result<()> {
    let bpe = BPE::from_file(vocab, merges).build()?;
    let mut tokenizer = Tokenizer::new(bpe);
    tokenizer
        .with_pre_tokenizer(ByteLevel::default())
        .with_decoder(ByteLevel::default());

    tokenizer.add_tokens(&[AddedToken::from(String::from("ing"), false).single_word(false)]);
    tokenizer
        .add_special_tokens(&[AddedToken::from(String::from("[ENT]"), true).single_word(true)]);

    let stdin = io::stdin();
    let mut handle = stdin.lock();
    let mut buffer = String::new();

    loop {
        buffer.clear();

        print!("\nEnter some text to tokenize:\n>  ");
        io::stdout().flush()?;
        handle.read_line(&mut buffer)?;
        let buffer = buffer.trim_end();

        let timer = std::time::Instant::now();
        let encoded = tokenizer.encode(buffer.to_owned(), false)?;
        let elapsed = timer.elapsed();
        println!("\nInput:\t\t{}", buffer);
        println!("Tokens:\t\t{:?}", encoded.get_tokens());
        println!("IDs:\t\t{:?}", encoded.get_ids());
        println!("Offsets:\t{:?}", encoded.get_offsets());
        println!(
            "Decoded:\t{}",
            tokenizer.decode(encoded.get_ids(), true).unwrap()
        );
        println!("Tokenized in {:?}", elapsed);
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    match args.command {
        Command::Shell { vocab, merges } => shell(&vocab, &merges),
    }
}
