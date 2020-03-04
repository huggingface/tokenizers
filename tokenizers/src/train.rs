use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;
use std::collections::HashMap;
use clap::{value_t, Arg, App};
use tokenizers::tokenizer::Model;
use tokenizers::models::bpe::BpeTrainer;

fn main() {
    let matches = App::new("BPE Trainer")
        .version("0.1.0")
        .author("Christian Hadiwinoto <christian.hadiwinoto@shopee.com>")
        .about("Training new BPE model from tab-separated word count file")
        .arg(
            Arg::with_name("word_count")
                .help("word count file")
                .index(1)
                .required(true),
        )
        .arg(
            Arg::with_name("output_dir")
                .help("output directory (must be precreated)")
                .index(2)
                .required(true),
        )
        .arg(
            Arg::with_name("special")
                .help("special tokens file, line-separated")
                .short("p")
                .long("special")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("vocab_size")
                .help("Desired vocab size")
                .takes_value(true)
                .short("s")
                .long("vocab_size")
                .takes_value(true)
                .default_value("30000")
        )
        .arg(
            Arg::with_name("min_freq")
                .help("Minimum frequency")
                .short("m")
                .long("min_freq")
                .takes_value(true)
                .default_value("2")
        )
        .get_matches();
    let wc_fname = matches.value_of("word_count").unwrap().to_string(); // files containing word count
    let out_dirname = matches.value_of("output_dir").unwrap().to_string(); // directory name to store output
    let vocab_size = value_t!(matches, "vocab_size", usize).unwrap_or(0);
    println!("| Vocabulary size is {}", vocab_size);
    let min_freq = value_t!(matches, "min_freq", u32).unwrap_or(0);
    println!("| Minimum frequency is {}", min_freq);
    
    // Instantiate trainer builder
    let trainer_builder = BpeTrainer::builder();
    let mut tokens: Vec<String> = Vec::new();
    let mut word_counts: HashMap<String, u32> = HashMap::new();
    
    if let Some(spc_fname) = matches.value_of("special") {
        if let Ok(lines) = read_lines(spc_fname) {
            // Read special vocab file, consumes the iterator, returns an (Optional) String
            for line in lines {
                if let Ok(tok) = line {
                    tokens.push(tok);
                }
            }
        }
    }

    // Train tokenizer from the word counts
    let trainer = trainer_builder
        .special_tokens(tokens)
        .vocab_size(vocab_size)
        .min_frequency(min_freq)
        .build();
    if let Ok(lines) = read_lines(wc_fname) {
        // Read word count file, consumes the iterator, returns an (Optional) String
        for line in lines {
            if let Ok(word_count_str) = line {
                let toks: Vec<&str> = word_count_str.split('\t').collect();
                let count: u32 = toks[1].parse().unwrap();
                word_counts.insert(toks[0].to_string(), count);
            }
        }
        let (model, special_tokens) = trainer.train(word_counts).unwrap();
        model.save(Path::new(&out_dirname), None);
    }
}

// The output is wrapped in a Result to allow matching on errors
// Returns an Iterator to the Reader of the lines of the file.
fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where P: AsRef<Path>, {
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}
