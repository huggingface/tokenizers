#[macro_use]
extern crate criterion;
use criterion::Criterion;
use std::hint::black_box;
use std::path::PathBuf;
use tokenizers::{normalizers::*, AddedToken, Normalizer, Tokenizer};

fn saved_tokenizer_path<N: Normalizer + Into<NormalizerWrapper>>(
    size: i64,
    normalizer: Option<N>,
    special_tokens: bool,
) -> PathBuf {
    let mut tokenizer = Tokenizer::from_pretrained("t5-small", None).unwrap();

    if let Some(norm) = normalizer {
        tokenizer.with_normalizer(Some(norm));
    }

    let tokens: Vec<_> = (0..size)
        .map(|i| AddedToken::from(format!("tok{i}"), special_tokens))
        .collect();
    tokenizer.add_tokens(tokens);

    let path = std::env::temp_dir().join(format!(
        "bench_tok_{size}_{special_tokens}_{}.json",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .subsec_nanos()
    ));
    tokenizer.save(&path, false).unwrap();
    path
}

#[allow(clippy::type_complexity)]
fn bench_deserialize(c: &mut Criterion) {
    let normalizers: Vec<(&str, Option<fn() -> NormalizerWrapper>)> =
        vec![("none", None), ("nfkc", Some(|| NFKC.into()))];

    for &size in &[100_000, 400_000] {
        for (norm_name, maybe_factory) in &normalizers {
            let label = format!(
                "special tokens deserialize_added_vocab_{}_norm_{}",
                size, norm_name
            );

            let path = match maybe_factory {
                Some(factory) => saved_tokenizer_path(size, Some(factory()), true),
                None => saved_tokenizer_path::<NormalizerWrapper>(size, None, true),
            };
            c.bench_function(&label, |b| {
                b.iter(|| {
                    let tok: Tokenizer = black_box(Tokenizer::from_file(&path).unwrap());
                    black_box(tok);
                })
            });
            std::fs::remove_file(&path).unwrap();

            let label = format!(
                "non special deserialize_added_vocab_{}_norm_{}",
                size, norm_name
            );

            let path = match maybe_factory {
                Some(factory) => saved_tokenizer_path(size, Some(factory()), false),
                None => saved_tokenizer_path::<NormalizerWrapper>(size, None, false),
            };
            c.bench_function(&label, |b| {
                b.iter(|| {
                    let tok: Tokenizer = black_box(Tokenizer::from_file(&path).unwrap());
                    black_box(tok);
                })
            });
            std::fs::remove_file(&path).unwrap();
        }
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_deserialize
}
criterion_main!(benches);
