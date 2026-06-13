//! Microbenchmark for the ASCII lowercase fast path used by the `Lowercase`
//! normalizer. Compares the SIMD-dispatched `utils::simd::ascii_lower` against
//! the scalar reference and against the previous Unicode-aware path
//! (`char::to_lowercase` per char) on representative buffer sizes.

#[macro_use]
extern crate criterion;

use criterion::{Criterion, Throughput};
use std::hint::black_box;
use tokenizers::utils::simd::ascii_lower;

fn make_buffer(len: usize) -> Vec<u8> {
    // Cycle through printable ASCII (mix of upper, lower, digits, punctuation)
    // so the upper-case branch fires on roughly 26/95 of bytes.
    (0..len).map(|i| 0x20u8 + (i as u8 % 0x5F)).collect()
}

fn scalar_lower(buf: &mut [u8]) {
    for b in buf {
        if b.is_ascii_uppercase() {
            *b |= 0x20;
        }
    }
}

fn unicode_lower(buf: &str) -> String {
    // Mirrors what `NormalizedString::lowercase` did before the fast path: per
    // `char` UTF-8 decode + Unicode case folding, ignoring alignment bookkeeping.
    let mut out = String::with_capacity(buf.len());
    for c in buf.chars() {
        for lc in c.to_lowercase() {
            out.push(lc);
        }
    }
    out
}

pub fn bench_ascii_lower(c: &mut Criterion) {
    for &len in &[64usize, 1024, 16 * 1024, 256 * 1024] {
        let mut group = c.benchmark_group(format!("ascii_lower/{len}B"));
        group.throughput(Throughput::Bytes(len as u64));

        let original = make_buffer(len);

        group.bench_function("simd", |b| {
            let mut buf = original.clone();
            b.iter(|| {
                ascii_lower(black_box(&mut buf));
            });
        });

        group.bench_function("scalar", |b| {
            let mut buf = original.clone();
            b.iter(|| {
                scalar_lower(black_box(&mut buf));
            });
        });

        // Stand-in for the pre-SIMD code path.
        let s = String::from_utf8(original.clone()).unwrap();
        group.bench_function("unicode_chars", |b| {
            b.iter(|| {
                black_box(unicode_lower(black_box(&s)));
            });
        });

        group.finish();
    }
}

criterion_group! {
    name = ascii_lower_bench;
    config = Criterion::default().sample_size(50);
    targets = bench_ascii_lower
}
criterion_main!(ascii_lower_bench);
