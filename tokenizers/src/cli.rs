//!
//! This is the CLI binary for the Tokenizers project
//!

use clap::{App, AppSettings, Arg, ArgMatches, SubCommand};
use std::io::{self, BufRead, Write};
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::tokenizer::{AddedToken, Result};
use tokenizers::Tokenizer;

fn shell(matches: &ArgMatches) -> Result<()> {
    let vocab = matches
        .value_of("vocab")
        .expect("Must give a vocab.json file");
    let merges = matches
        .value_of("merges")
        .expect("Must give a merges.txt file");

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
            tokenizer.decode(encoded.get_ids().to_vec(), true).unwrap()
        );
        println!("Tokenized in {:?}", elapsed);
    }
}

fn main() -> Result<()> {
    let matches = App::new("tokenizers")
        .version("0.0.1")
        .author("Anthony M. <anthony@huggingface.co>")
        .about("Generate custom Tokenizers or use existing ones")
        .setting(AppSettings::SubcommandRequiredElseHelp)
        .subcommand(
            SubCommand::with_name("shell")
                .about("Interactively test a tokenizer")
                .arg(
                    Arg::with_name("vocab")
                        .long("vocab")
                        .value_name("VOCAB_FILE")
                        .help("Path to the vocab.json file")
                        .required(true),
                )
                .arg(
                    Arg::with_name("merges")
                        .long("merges")
                        .value_name("MERGES_FILE")
                        .help("Path to the merges.txt file")
                        .required(true),
                ),
        )
        .get_matches();

    match matches.subcommand() {
        ("shell", matches) => shell(matches.unwrap()),
        (subcommand, _) => panic!("Unknown subcommand {}", subcommand),
    }
}
