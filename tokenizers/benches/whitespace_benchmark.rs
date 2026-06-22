#[macro_use]
extern crate criterion;

use criterion::{Criterion, Throughput};
use tokenizers::pre_tokenizers::whitespace::{Whitespace, WhitespaceOptimized};
use tokenizers::{OffsetReferential, OffsetType, PreTokenizedString, PreTokenizer};

fn bench_whitespace_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("whitespace-pre-tokenizers");

    // Test data with various characteristics
    let test_cases = vec![
        ("simple", "Hello world! How are you doing?"),
        (
            "mixed",
            "This is a test with numbers 123 and symbols @#$% and unicode: café résumé",
        ),
        (
            "whitespace_heavy",
            "Multiple    spaces\tand\nnewlines\r\nhere",
        ),
        ("symbol_heavy", "Hello!@#$%^&*()world?><>{}[]|\\"),
        (
            "word_heavy",
            "This is a very long sentence with many words that should be tokenized properly",
        ),
        ("unicode_heavy", "αβγ δέζ ηθι κλμ νξο πρσ τυφ χψω"),
        ("mixed_unicode", "Hello 123 αβγ !@# world δέζ ηθι"),
    ];

    for (name, text) in test_cases {
        let data_len = text.len() as u64;
        group.throughput(Throughput::Bytes(data_len));

        // Benchmark original regex-based implementation
        group.bench_function(format!("{}-original", name), |b| {
            b.iter(|| {
                let mut pretokenized = PreTokenizedString::from(text);
                let pretok = Whitespace {};
                pretok.pre_tokenize(&mut pretokenized).unwrap();
                let _result = pretokenized
                    .get_splits(OffsetReferential::Original, OffsetType::Byte)
                    .into_iter()
                    .map(|(s, o, _)| (s, o))
                    .collect::<Vec<_>>();
            })
        });

        // Benchmark optimized byte-level implementation
        group.bench_function(format!("{}-optimized", name), |b| {
            b.iter(|| {
                let mut pretokenized = PreTokenizedString::from(text);
                let pretok = WhitespaceOptimized {};
                pretok.pre_tokenize(&mut pretokenized).unwrap();
                let _result = pretokenized
                    .get_splits(OffsetReferential::Original, OffsetType::Byte)
                    .into_iter()
                    .map(|(s, o, _)| (s, o))
                    .collect::<Vec<_>>();
            })
        });
    }

    group.finish();
}

fn bench_large_text(c: &mut Criterion) {
    let mut group = c.benchmark_group("whitespace-large-text");

    // Create a large text by repeating patterns
    let base_text =
        "Hello world! This is a test with numbers 123 and symbols @#$% and unicode: café résumé. ";
    let large_text: String = base_text.repeat(1000); // ~50KB of text
    let data_len = large_text.len() as u64;

    group.throughput(Throughput::Bytes(data_len));

    group.bench_function("large-original", |b| {
        b.iter(|| {
            let mut pretokenized = PreTokenizedString::from(large_text.as_str());
            let pretok = Whitespace {};
            pretok.pre_tokenize(&mut pretokenized).unwrap();
            let _result = pretokenized
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>();
        })
    });

    group.bench_function("large-optimized", |b| {
        b.iter(|| {
            let mut pretokenized = PreTokenizedString::from(large_text.as_str());
            let pretok = WhitespaceOptimized {};
            pretok.pre_tokenize(&mut pretokenized).unwrap();
            let _result = pretokenized
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>();
        })
    });

    group.finish();
}

criterion_group! {
    name = whitespace_benches;
    config = Criterion::default().sample_size(20);
    targets = bench_whitespace_comparison, bench_large_text
}

criterion_main!(whitespace_benches);
