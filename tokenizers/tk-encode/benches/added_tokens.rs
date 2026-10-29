//! Added-token search throughput: [`Buckets::match_bytes`] driven over a whole input, the way the
//! encoder splits added tokens out of a text.
//!
//! Inputs are synthetic so the share of candidate bytes (bytes that start an added token) is exact.
//! tokbench's real corpora reach about 1.6%.
//!
//! Compare a branch with the main it forked from, from one checkout, in the `tokenizers/` workspace
//! directory (only `buckets.rs` differs):
//!
//! ```sh
//! git checkout $(git merge-base HEAD origin/main) -- tk-encode/src/vocab/buckets.rs
//! cargo bench -p tk-encode --bench added_tokens -- --save-baseline main
//! git checkout HEAD -- tk-encode/src/vocab/buckets.rs
//! cargo bench -p tk-encode --bench added_tokens -- --baseline main
//! ```
//!
//! On x86_64 main scans byte by byte, so this measures the SSSE3 scan. On aarch64 main already
//! has a NEON scan, so it measures the refactor only.

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use tk_encode::vocab::buckets::Buckets;

const INPUT_LEN: usize = 4_000_000;

fn vocab(tokens: &[&str]) -> Buckets {
    let mut pairs = Vec::new();
    for (id, token) in tokens.iter().enumerate() {
        pairs.push((token.as_bytes().to_vec(), id as u32));
    }
    Buckets::from_tokens(pairs).unwrap()
}

/// Calls `match_bytes` from the start, then again past each match, until the end.
fn find_all(buckets: &Buckets, bytes: &[u8]) -> usize {
    let (mut matches, mut search) = (0, 0);
    while let Some((_, start, len)) = buckets.match_bytes(&bytes[search..]) {
        matches += 1;
        search += (start + len) as usize;
    }
    matches
}

/// `a` filler where `percent` of the bytes are a false candidate `<`, plus a real `<|eos|>`
/// every 10 kB.
fn with_false_candidates(percent: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(INPUT_LEN + 8);
    let mut acc = 0;
    while out.len() < INPUT_LEN {
        acc += percent;
        if acc >= 100 {
            acc -= 100;
            out.push(b'<');
        } else {
            out.push(b'a');
        }
        if out.len() % 10_000 == 0 {
            out.extend_from_slice(b"<|eos|>");
        }
    }
    out
}

/// `a` filler with `token` every `every` bytes.
fn with_token_every(every: usize, token: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(INPUT_LEN + every);
    while out.len() < INPUT_LEN {
        out.extend(std::iter::repeat_n(b'a', every - token.len()));
        out.extend_from_slice(token);
    }
    out
}

fn bench(c: &mut Criterion) {
    // Two distinct first bytes (`<`, `[`): the nibble scan.
    let five = vocab(&["<|bos|>", "<|eos|>", "<|pad|>", "[CLS]", "[SEP]"]);
    // One first byte: the memmem search for the shared prefix `<|`.
    let single = vocab(&["<|bos|>", "<|eos|>", "<|pad|>"]);

    let mut group = c.benchmark_group("candidate_bytes");
    for percent in [1, 5, 25, 50, 90] {
        let input = with_false_candidates(percent);
        group.throughput(Throughput::Bytes(input.len() as u64));
        group.bench_function(format!("{percent}%"), |b| {
            b.iter(|| find_all(&five, black_box(&input)))
        });
    }
    group.finish();

    let mut group = c.benchmark_group("token_every");
    for every in [16, 64, 256] {
        let input = with_token_every(every, b"<|eos|>");
        group.throughput(Throughput::Bytes(input.len() as u64));
        group.bench_function(format!("{every}B/five_tokens"), |b| {
            b.iter(|| find_all(&five, black_box(&input)))
        });
        group.bench_function(format!("{every}B/single_bucket"), |b| {
            b.iter(|| find_all(&single, black_box(&input)))
        });
    }
    group.finish();

    // `<|` with no match behind it: every hit of the single bucket's prefix is a false candidate.
    let mut group = c.benchmark_group("false_prefix_every");
    for every in [16, 64] {
        let input = with_token_every(every, b"<|xs0|>");
        group.throughput(Throughput::Bytes(input.len() as u64));
        group.bench_function(format!("{every}B/single_bucket"), |b| {
            b.iter(|| find_all(&single, black_box(&input)))
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
