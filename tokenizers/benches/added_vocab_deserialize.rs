#[macro_use]
extern crate criterion;
use criterion::Criterion;
use std::hint::black_box;
use std::{process::exit, str::FromStr};
use tokenizers::{normalizers::*, AddedToken, Normalizer, Tokenizer};

fn serialized_tokenizer<N: Normalizer + Into<NormalizerWrapper>>(
    size: i64,
    normalizer: Option<N>,
    special_tokens: bool,
) -> String {
    let mut tokenizer = Tokenizer::from_pretrained("t5-small", None).unwrap();

    if let Some(norm) = normalizer {
        tokenizer.with_normalizer(Some(norm));
    }

    let tokens: Vec<_> = (0..size)
        .map(|i| AddedToken::from(format!("tok{i}"), special_tokens))
        .collect();
    tokenizer.add_tokens(&tokens);

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

    for &size in &[100_000, 400_000] {
        for (norm_name, maybe_factory) in &normalizers {
            let label = format!(
                "special tokens deserialize_added_vocab_{}_norm_{}",
                size, norm_name
            );

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

            let label = format!(
                "non special deserialize_added_vocab_{}_norm_{}",
                size, norm_name
            );

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

criterion_group! {
    name = benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_deserialize
}
criterion_main!(benches);
