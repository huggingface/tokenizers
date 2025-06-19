use tokenizers::models::wordpiece::WordPiece;
use tokenizers::{AddedToken, Tokenizer};

fn main() {
    let start = std::time::Instant::now();
    let mut tokenizer = Tokenizer::new(WordPiece::default());

    // Mix special and not special
    // You can make sure ids are in order, and special status is correct.
    let tokens: Vec<_> = (0..120_000)
        .map(|i| AddedToken::from(format!("[SPECIAL_{i}]"), i % 2 == 0))
        .collect();
    tokenizer.add_tokens(&tokens);
    tokenizer.save("_tok.json", true).unwrap();
    println!("Save took {:?}", start.elapsed());
    let start = std::time::Instant::now();
    let _tok = Tokenizer::from_file("_tok.json").unwrap();
    println!("Took {:?}", start.elapsed());
    std::fs::remove_file("_tok.json").unwrap();
}
