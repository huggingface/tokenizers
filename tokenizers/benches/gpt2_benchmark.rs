#[macro_use]
extern crate criterion;

use criterion::black_box;
use criterion::Criterion;
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::tokenizer::{EncodeInput, Tokenizer};

fn gpt2_benchmark(c: &mut Criterion) {
    let bpe = BPE::from_files("gpt2-vocab.json", "gpt2-merges.txt").unwrap();
    let mut tokenizer = Tokenizer::new(Box::new(bpe));
    tokenizer.with_pre_tokenizer(Box::new(ByteLevel::new(false)));
    let sentence = "The quick brown fox jumps over the lazy dog";

    c.bench_function("gpt2 encode", |b| {
        b.iter(|| tokenizer.encode(black_box(EncodeInput::Single(String::from(sentence)))))
    });
}

criterion_group!(benches, gpt2_benchmark);
criterion_main!(benches);
