#[macro_use]
extern crate criterion;
use criterion::Criterion;
use tokenizers::models::wordpiece::WordPiece;
use tokenizers::{AddedToken, Tokenizer};

fn bench_load(c: &mut Criterion) {
    let mut tokenizer = Tokenizer::new(WordPiece::default());

    let tokens: Vec<_> = (0..4000)
        .map(|i| AddedToken::from(format!("[SPECIAL_{}]", i), false))
        .collect();
    tokenizer.add_tokens(&tokens);
    tokenizer.save("_tok.json", false).unwrap();
    c.bench_function("Loading tokenizer with many saved tokens", |b| {
        b.iter(|| Tokenizer::from_file("_tok.json").unwrap())
    });
}
criterion_group! {
    name = benches_load;
    config = Criterion::default().sample_size(10);
    targets = bench_load
}
criterion_main!(benches_load);
