#[macro_use]
extern crate criterion;
use std::{process::exit, str::FromStr};
use criterion::{black_box, Criterion};
use std::collections::HashMap;
use tokenizers::{models::wordlevel::WordLevel, normalizers::*, AddedToken, Normalizer, Tokenizer};

fn serialized_tokenizer<N: Normalizer + Into<NormalizerWrapper>>(
    size: usize,
    normalizer: Option<N>,
    special_tokens: bool,
) -> String {
    let mut vocab = HashMap::new();
    vocab.insert("a".to_string(), 0);
    let model = WordLevel::builder()
        .vocab(vocab)
        .unk_token("[UNK]".into())
        .build()
        .unwrap();

    let mut tokenizer = Tokenizer::new(model);
    let tokens: Vec<_> = (0..size)
        .map(|i| AddedToken::from(format!("tok{i}"), special_tokens))
        .collect();
    tokenizer.add_tokens(&tokens);

    if let Some(norm) = normalizer {
        tokenizer.with_normalizer(Some(norm));
    }

    serde_json::to_string(&tokenizer).unwrap()
}

fn bench_deserialize(c: &mut Criterion) {
    let normalizers: Vec<(&str, Option<fn() -> NormalizerWrapper>)> = vec![
        ("none", None),
        ("byte_level", Some(|| ByteLevel::default().into())),
        ("lowercase", Some(|| Lowercase.into())),
        ("nfc", Some(|| NFC.into())),
        ("nfd", Some(|| NFD.into())),
        ("nfkc", Some(|| NFKC.into())),
        ("nfkd", Some(|| NFKD.into())),
        ("nmt", Some(|| Nmt.into())),
        ("strip", Some(|| Strip::new(true, true).into())),
        ("replace", Some(|| Replace::new("a", "b").unwrap().into())),
        ("prepend", Some(|| Prepend::new("pre_".to_string()).into())),
        ("bert", Some(|| BertNormalizer::default().into())),
    ];

    for &size in &[10_000usize, 100_000, 400_000] {
        for (norm_name, maybe_factory) in &normalizers {
            let label = format!("special tokens deserialize_added_vocab_{}_norm_{}", size, norm_name);

            let json = match maybe_factory {
                Some(factory) => serialized_tokenizer(size, Some(factory()), true),
                None => serialized_tokenizer::<NormalizerWrapper>(size, None, true),
            };
            c.bench_function(&label, |b| {
                b.iter(|| {
                    let tok: Tokenizer = black_box(Tokenizer::from_str(&json).unwrap());
                    black_box(tok);
                })
            });

            let label = format!("non special deserialize_added_vocab_{}_norm_{}", size, norm_name);

            let json = match maybe_factory {
                Some(factory) => serialized_tokenizer(size, Some(factory()), false),
                None => serialized_tokenizer::<NormalizerWrapper>(size, None, false),
            };
            c.bench_function(&label, |b| {
                b.iter(|| {
                    let tok: Tokenizer = black_box(Tokenizer::from_str(&json).unwrap());
                    black_box(tok);
                })
            });
        }
        exit(0);
    }
}

criterion_group!(benches, bench_deserialize);
criterion_main!(benches);
