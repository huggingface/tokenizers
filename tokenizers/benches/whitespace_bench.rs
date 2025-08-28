use std::hint::black_box;
use criterion::{criterion_group, criterion_main, Criterion};
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::tokenizer::PreTokenizedString;
use tokenizers::PreTokenizer;

fn bench_whitespace(c: &mut Criterion) {
    let tokenizer = Whitespace;

    let short_text = "Hello world!";
    let medium_text = "This is a sentence with    multiple spaces. And\tsome\nnewlines. Also, punctuation! Like.this? And unicode: こんにちは世界 🙏🏽";
    let long_text = "The quick brown fox jumps over the lazy dog. This is a much longer piece of text designed to test the performance of the whitespace pre-tokenizer on a substantial input. It includes various forms of whitespace, such as multiple consecutive spaces, tabs\tbetween words, and newlines.\nIt also mixes in different types of characters, including numbers (123), special symbols (!@#$%^&*()), and some common emojis like 😂👍✨. The goal is to ensure that the tokenizer correctly splits the text into tokens while maintaining its performance characteristics across a diverse set of linguistic features. This paragraph will be repeated several times to create a truly long input for comprehensive benchmarking. The quick brown fox jumps over the lazy dog. This is a much longer piece of text designed to test the performance of the whitespace pre-tokenizer on a substantial input. It includes various forms of whitespace, such as multiple consecutive spaces, tabs\tbetween words, and newlines.\nIt also mixes in different types of characters, including numbers (123), special symbols (!@#$%^&*()), and some common emojis like 😂👍✨. The goal is to ensure that the tokenizer correctly splits the text into tokens while maintaining its performance characteristics across a diverse set of linguistic features. This paragraph will be repeated several times to create a truly long input for comprehensive benchmarking. The quick brown fox jumps over the lazy dog. This is a much longer piece of text designed to test the performance of the whitespace pre-tokenizer on a substantial input. It includes various forms of whitespace, such as multiple consecutive spaces, tabs\tbetween words, and newlines.\nIt also mixes in different types of characters, including numbers (123), special symbols (!@#$%^&*()), and some common emojis like 😂👍✨. The goal is to ensure that the tokenizer correctly splits the text into tokens while maintaining its performance characteristics across a diverse set of linguistic features.";

    let samples = vec![
        ("short_unique", short_text),
        ("medium_unique", medium_text),
        ("long_unique", long_text),
    ];

    for (label, text) in samples {
        c.bench_function(&format!("whitespace_pretokenizer_{}", label), |b| {
            b.iter(|| {
                let mut s = PreTokenizedString::from(black_box(text));
                tokenizer.pre_tokenize(&mut s).unwrap();
                black_box(&s);
            });
        });
    }
}

criterion_group!(benches, bench_whitespace);
criterion_main!(benches);
