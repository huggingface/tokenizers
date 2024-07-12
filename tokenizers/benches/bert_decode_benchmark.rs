use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tokenizers::tokenizer::Tokenizer;

fn decode(tokenizer: &Tokenizer, ids_slice: Vec<u32>, skip_special_tokens: bool) -> String {
    tokenizer
        .decode(ids_slice, skip_special_tokens)
        .expect("failed to decode input")
}

fn criterion_benchmark(c: &mut Criterion) {
    let tokenizer =
        Tokenizer::from_file("data/bert-wiki.json").expect("failed to create tokenizer");
    c.bench_function("decode", |b| {
        b.iter(|| {
            decode(
                &tokenizer,
                black_box([2829, 4419, 14523, 2058, 1996, 13971, 3899].to_vec()),
                black_box(true),
            )
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
