use clap::{Parser, Subcommand};
use tokenizers::tokenizer::Tokenizer;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Tokenize input text using a model file
    Tokenize {
        /// Path to the tokenizer model file (e.g., tokenizer.json)
        #[arg(long)]
        model: String,
        /// Input text to tokenize
        #[arg(long)]
        text: String,
    },
    /// Train a new BPE tokenizer model
    Train {
        /// Input text file(s) for training (comma-separated or repeated)
        #[arg(long, required = true)]
        files: Vec<String>,
        /// Vocabulary size
        #[arg(long, default_value_t = 30000)]
        vocab_size: usize,
        /// Output path for the trained model (e.g., model.json)
        #[arg(long)]
        output: String,
    },
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Tokenize { model, text } => match Tokenizer::from_file(&model) {
            Ok(tokenizer) => match tokenizer.encode(text.as_str(), true) {
                Ok(encoding) => {
                    println!("Token IDs: {:?}", encoding.get_ids());
                }
                Err(e) => {
                    eprintln!("Failed to encode text: {}", e);
                    std::process::exit(1);
                }
            },
            Err(e) => {
                eprintln!("Failed to load tokenizer model: {}", e);
                std::process::exit(1);
            }
        },
        Commands::Train {
            files,
            vocab_size,
            output,
        } => {
            use tokenizers::models::bpe::{BpeTrainer, BPE};
            use tokenizers::models::ModelWrapper;
            use tokenizers::models::TrainerWrapper;

            let mut tokenizer = Tokenizer::new(ModelWrapper::BPE(BPE::default()));
            let mut trainer =
                TrainerWrapper::BpeTrainer(BpeTrainer::builder().vocab_size(vocab_size).build());
            if let Err(e) = tokenizer.train_from_files(&mut trainer, files.clone()) {
                eprintln!("Failed to train tokenizer: {}", e);
                std::process::exit(1);
            }
            if let Err(e) = tokenizer.save(&output, true) {
                eprintln!("Failed to save trained model: {}", e);
                std::process::exit(1);
            }
            println!("Model trained and saved to {}", output);
        }
    }
}
