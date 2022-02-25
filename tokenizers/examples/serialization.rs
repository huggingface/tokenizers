use tokenizers::models::wordpiece::WordPiece;
use tokenizers::{AddedToken, Tokenizer};

fn main() {
    let start = std::time::Instant::now();
    let mut tokenizer = Tokenizer::new(WordPiece::default());

    let tokens: Vec<_> = (0..120_000)
        .map(|i| AddedToken::from(format!("[SPECIAL_{}]", i), false))
        .collect();
    tokenizer.add_tokens(&tokens);
    tokenizer.save("_tok.json", false).unwrap();
    println!("Save took {:?}", start.elapsed());
    let start = std::time::Instant::now();
    let _tok = Tokenizer::from_file("_tok.json").unwrap();
    println!("Took {:?}", start.elapsed());
    std::fs::remove_file("_tok.json").unwrap();
}
