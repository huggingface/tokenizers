//!
//! This is the CLI binary for the Tokenizers project
//!

use clap::{App, AppSettings, Arg, ArgMatches, SubCommand};
use std::io::{self, BufRead, Write};
use tokenizers::models::bpe::{Error, BPE};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::tokenizer::Tokenizer;

fn shell(matches: &ArgMatches) -> Result<(), Error> {
    let vocab = matches
        .value_of("vocab")
        .expect("Must give a vocab.json file");
    let merges = matches
        .value_of("merges")
        .expect("Must give a merges.txt file");

    let bpe = BPE::from_files(vocab, merges)?;
    let mut tokenizer = Tokenizer::new(Box::new(bpe));
    tokenizer.with_pre_tokenizer(Box::new(ByteLevel));

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
        let encoded = tokenizer.encode(buffer);
        let elapsed = timer.elapsed();
        println!("\nInput:\t\t{}", buffer);
        println!(
            "Tokens:\t\t{:?}",
            encoded.iter().map(|t| &t.value).collect::<Vec<_>>()
        );
        println!(
            "IDs:\t\t{:?}",
            encoded.iter().map(|t| t.id).collect::<Vec<_>>()
        );
        println!(
            "Offsets:\t{:?}",
            encoded.iter().map(|t| t.offsets).collect::<Vec<_>>()
        );
        println!("Tokenized in {:?}", elapsed);
    }
}

fn main() -> Result<(), Error> {
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
