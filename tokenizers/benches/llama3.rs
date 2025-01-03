#[macro_use]
extern crate criterion;

use criterion::{Criterion, Throughput};
use itertools::Itertools;
use tokenizers::models::backtracking_bpe;
use tokenizers::Tokenizer;

pub fn llama3(c: &mut Criterion) {
    let data = std::fs::read_to_string("data/big.txt").unwrap();
    let mut group = c.benchmark_group("llama3-encode");
    group.throughput(Throughput::Bytes(data.bytes().len() as u64));

    group.bench_function("llama3-backtracking", |b| {
        let tokenizer = Tokenizer::from_pretrained("gpt2", None).unwrap();
        let mut vocab = &mut tokenizer.get_vocab(false).clone().into_iter().collect::<Vec<_>>(); // Convert HashMap into a Vec of (String, u32) tuples
                                                                                    //
        vocab.sort_by(|a, b| a.1.cmp(&b.1));
        vocab.truncate(vocab.len().saturating_sub(3));
        let vocab: Vec<_>  = vocab // Sort by u32 value
            .into_iter() // IntoIterator to get the iterator of Vec<u8>
            .map(|(tok, _)| Vec::from(tok.as_bytes()))
            .collect();
        let model: backtracking_bpe::BacktrackingBpe =
            backtracking_bpe::BacktrackingBpe::from_dictionary(vocab, None, None);
        let tokenizer = Tokenizer::new(model);
        let data: Vec<_> = data.lines().collect();
        let add_special_tokens = false;
        b.iter(|| {
            tokenizer
                .encode_batch(criterion::black_box(data.clone()), add_special_tokens)
                .unwrap()
        })
    });

    group.bench_function("llama3-offsets", |b| {
        let tokenizer = Tokenizer::from_pretrained("gpt2", None).unwrap();
        let data: Vec<_> = data.lines().collect();
        let add_special_tokens = false;
        b.iter(|| {
            tokenizer
                .encode_batch_char_offsets(criterion::black_box(data.clone()), add_special_tokens)
                .unwrap()
        })
    });
    group.bench_function("llama3-nooffsets", |b| {
        let tokenizer = Tokenizer::from_pretrained("gpt2", None).unwrap();
        let data: Vec<_> = data.lines().collect();
        let add_special_tokens = false;
        b.iter(|| {
            tokenizer
                .encode_batch(criterion::black_box(data.clone()), add_special_tokens)
                .unwrap()
        })
    });

    group.finish();
}

criterion_group! {
    name = llama;
    config = Criterion::default().sample_size(10);
    targets = llama3
}

criterion_main!(llama);
