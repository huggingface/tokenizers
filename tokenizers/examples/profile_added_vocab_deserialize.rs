use std::path::Path;
use tokenizers::{normalizers::NormalizerWrapper, AddedToken, Tokenizer};

fn main() {
    // Build tokenizer: t5-small base, strip its normalizer (None), 100_000 special added tokens
    let mut tokenizer = Tokenizer::from_pretrained("t5-small", None).unwrap();
    tokenizer.with_normalizer(None::<NormalizerWrapper>);

    let tokens: Vec<_> = (0..100_000)
        .map(|i| AddedToken::from(format!("tok{i}"), true))
        .collect();

    std::hint::black_box(tokenizer.add_tokens(tokens));

    let path = Path::new("/tmp/profile_added_vocab.json");
    tokenizer.save(path, false).unwrap();
    println!("Saved tokenizer to {}", path.display());

    // Tight loop so samply captures meaningful samples
    for _ in 0..20 {
        let _tok = std::hint::black_box(Tokenizer::from_file(path).unwrap());
    }

    std::fs::remove_file(path).unwrap();
    println!("Done.");
}
