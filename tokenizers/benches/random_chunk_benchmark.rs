#[macro_use]
extern crate criterion;

mod common;

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::{Duration, Instant};

use criterion::{black_box, Criterion};
use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::models::TrainerWrapper;
use tokenizers::pre_tokenizers::random_chunk::RandomChunkSplit;
use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;
use tokenizers::tokenizer::{PreTokenizedString, PreTokenizer, TokenizerImpl};
use tokenizers::{OffsetReferential, OffsetType, Tokenizer};
use tokenizers::models::ModelWrapper;
use tokenizers::normalizers::NormalizerWrapper;
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::processors::PostProcessorWrapper;
use tokenizers::decoders::DecoderWrapper;

use common::iter_bench_train;

// Benchmark different RandomChunkSplit configurations for pre-tokenization
fn bench_pre_tokenize(c: &mut Criterion) {
    // Load sample text for benchmarking
    let mut text = String::new();
    for line in BufReader::new(File::open(Path::new("data/small.txt")).unwrap())
        .lines()
        .take(100)
    {
        text.push_str(&line.unwrap());
        text.push(' ');
    }

    // Create a benchmark group for pre-tokenization
    let mut group = c.benchmark_group("PreTokenization");

    // Benchmark WhitespaceSplit as baseline
    let whitespace_split = WhitespaceSplit;
    group.bench_function("WhitespaceSplit", |b| {
        b.iter_custom(|iters| {
            let mut duration = Duration::new(0, 0);
            for _ in 0..iters {
                let mut pretokenized = PreTokenizedString::from(text.clone());
                let start = Instant::now();
                let _ = black_box(whitespace_split.pre_tokenize(&mut pretokenized));
                duration = duration.checked_add(start.elapsed()).unwrap();
            }
            duration
        });
    });

    // Benchmark different RandomChunkSplit configurations
    let configs = vec![
        (1, 3, "RandomChunkSplit(1,3)"),
        (1, 5, "RandomChunkSplit(1,5)"),
        (1, 10, "RandomChunkSplit(1,10)"),
        (2, 5, "RandomChunkSplit(2,5)"),
        (3, 7, "RandomChunkSplit(3,7)"),
        (5, 10, "RandomChunkSplit(5,10)"),
    ];

    for (min_len, max_len, name) in configs {
        let random_chunk_split = RandomChunkSplit::new(min_len, max_len);
        group.bench_function(name, |b| {
            b.iter_custom(|iters| {
                let mut duration = Duration::new(0, 0);
                for _ in 0..iters {
                    let mut pretokenized = PreTokenizedString::from(text.clone());
                    let start = Instant::now();
                    let _ = black_box(random_chunk_split.pre_tokenize(&mut pretokenized));
                    duration = duration.checked_add(start.elapsed()).unwrap();
                }
                duration
            });
        });
    }

    group.finish();
}

// Benchmark token statistics (average length, whitespace percentage, etc.)
fn bench_token_statistics(c: &mut Criterion) {
    // Load sample text for analysis
    let mut text = String::new();
    for line in BufReader::new(File::open(Path::new("data/small.txt")).unwrap())
        .lines()
        .take(100)
    {
        text.push_str(&line.unwrap());
        text.push(' ');
    }

    // Create a benchmark group for token statistics
    let mut group = c.benchmark_group("TokenStatistics");
    group.sample_size(50); // Reduce number of samples for this analysis

    // Analyze WhitespaceSplit (baseline)
    let whitespace_split = WhitespaceSplit;
    group.bench_function("WhitespaceSplit_Stats", |b| {
        b.iter_custom(|iters| {
            let mut duration = Duration::new(0, 0);
            for _ in 0..iters {
                let mut pretokenized = PreTokenizedString::from(text.clone());
                whitespace_split.pre_tokenize(&mut pretokenized).unwrap();
                
                let start = Instant::now();
                let splits = pretokenized.get_splits(OffsetReferential::Original, OffsetType::Char);
                
                // Calculate token statistics
                let total_tokens = splits.len();
                let total_chars: usize = splits.iter().map(|(s, _, _)| s.chars().count()).sum();
                let _avg_token_length = if total_tokens > 0 {
                    total_chars as f64 / total_tokens as f64
                } else {
                    0.0
                };
                
                // Count tokens with whitespace
                let tokens_with_whitespace = splits
                    .iter()
                    .filter(|(s, _, _)| s.contains(char::is_whitespace))
                    .count();
                let _whitespace_percentage = if total_tokens > 0 {
                    tokens_with_whitespace as f64 / total_tokens as f64 * 100.0
                } else {
                    0.0
                };
                
                black_box((_avg_token_length, _whitespace_percentage));
                duration = duration.checked_add(start.elapsed()).unwrap();
            }
            duration
        });
    });

    // Analyze different RandomChunkSplit configurations
    let configs = vec![
        (1, 3, "RandomChunkSplit(1,3)_Stats"),
        (1, 5, "RandomChunkSplit(1,5)_Stats"),
        (2, 5, "RandomChunkSplit(2,5)_Stats"),
        (5, 10, "RandomChunkSplit(5,10)_Stats"),
    ];

    for (min_len, max_len, name) in configs {
        let random_chunk_split = RandomChunkSplit::new(min_len, max_len);
        group.bench_function(name, |b| {
            b.iter_custom(|iters| {
                let mut duration = Duration::new(0, 0);
                for _ in 0..iters {
                    let mut pretokenized = PreTokenizedString::from(text.clone());
                    random_chunk_split.pre_tokenize(&mut pretokenized).unwrap();
                    
                    let start = Instant::now();
                    let splits = pretokenized.get_splits(OffsetReferential::Original, OffsetType::Char);
                    
                    // Calculate token statistics
                    let total_tokens = splits.len();
                    let total_chars: usize = splits.iter().map(|(s, _, _)| s.chars().count()).sum();
                    let _avg_token_length = if total_tokens > 0 {
                        total_chars as f64 / total_tokens as f64
                    } else {
                        0.0
                    };
                    
                    // Count tokens with whitespace
                    let tokens_with_whitespace = splits
                        .iter()
                        .filter(|(s, _, _)| s.contains(char::is_whitespace))
                        .count();
                    let _whitespace_percentage = if total_tokens > 0 {
                        tokens_with_whitespace as f64 / total_tokens as f64 * 100.0
                    } else {
                        0.0
                    };
                    
                    black_box((_avg_token_length, _whitespace_percentage));
                    duration = duration.checked_add(start.elapsed()).unwrap();
                }
                duration
            });
        });
    }

    group.finish();
}

// Benchmark model training with different pre-tokenizers
fn bench_train_with_different_pretok(c: &mut Criterion) {
    let mut group = c.benchmark_group("TrainingWithDifferentPreTokenizers");
    group.sample_size(10); // Training is expensive, use fewer samples

    type TokenizerType = TokenizerImpl<ModelWrapper, NormalizerWrapper, PreTokenizerWrapper, PostProcessorWrapper, DecoderWrapper>;
    
    // Using enum to store different types
    let configs = vec![
        ("WhitespaceSplit_Train", Box::new(|t: &mut TokenizerType| {
            t.with_pre_tokenizer(Some(WhitespaceSplit {}));
        }) as Box<dyn Fn(&mut TokenizerType)>),
        ("RandomChunkSplit(1,3)_Train", Box::new(|t: &mut TokenizerType| {
            t.with_pre_tokenizer(Some(RandomChunkSplit::new(1, 3)));
        })),
        ("RandomChunkSplit(2,5)_Train", Box::new(|t: &mut TokenizerType| {
            t.with_pre_tokenizer(Some(RandomChunkSplit::new(2, 5)));
        })),
        ("RandomChunkSplit(5,10)_Train", Box::new(|t: &mut TokenizerType| {
            t.with_pre_tokenizer(Some(RandomChunkSplit::new(5, 10)));
        })),
    ];

    for (name, setup_fn) in configs {
        let mut trainer: TrainerWrapper = BpeTrainerBuilder::default()
            .show_progress(false)
            .vocab_size(1000) // Smaller vocab for benchmark
            .min_frequency(2)
            .build()
            .into();
        
        let mut tokenizer = Tokenizer::new(BPE::default()).into_inner();
        setup_fn(&mut tokenizer);
        
        group.bench_function(name, |b| {
            b.iter_custom(|iters| {
                iter_bench_train(
                    iters,
                    &mut tokenizer,
                    &mut trainer,
                    vec!["data/small.txt".to_string()],
                )
            });
        });
    }

    group.finish();
}

criterion_group! {
    name = random_chunk_benches;
    config = Criterion::default().sample_size(30);
    targets = bench_pre_tokenize, bench_token_statistics
}

criterion_group! {
    name = random_chunk_train_benches;
    config = Criterion::default().sample_size(10);
    targets = bench_train_with_different_pretok
}

criterion_main!(random_chunk_benches, random_chunk_train_benches);