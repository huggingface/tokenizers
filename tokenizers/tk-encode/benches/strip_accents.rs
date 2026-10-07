//! `StripAccents` throughput, to compare the implementations of its Unicode 9 mark set.
//!
//! The set is picked at compile time by a feature on tk-encode, so run once per variant. Each
//! run is reported under its own name in every group:
//!
//! ```text
//! cargo bench --bench strip_accents --features normalizers
//! cargo bench --bench strip_accents --features normalizers,marks-set-nohash
//! cargo bench --bench strip_accents --features normalizers,marks-set-bitset
//! ```

use std::hint::black_box;

use criterion::{Criterion, SamplingMode, Throughput, criterion_group, criterion_main};
use tk_encode::normalizers::{NFD, StripAccents};
use tk_encode::pipeline::Normalizer;

const VARIANT: &str = if cfg!(feature = "marks-set-bitset") {
    "bitset"
} else if cfg!(feature = "marks-set-nohash") {
    "nohash"
} else {
    "fxhash"
};

/// Line length for the synthetic corpus, close to the fixtures' average line.
const SYNTHETIC_LINE_CHARS: usize = 64;

/// The language fixtures, NFD'd line by line. StripAccents runs after NFD in real configs, which
/// is what splits an accented letter into a base letter and a combining mark.
fn languages() -> Vec<(String, Vec<String>)> {
    let mut paths: Vec<_> = std::fs::read_dir("../data/fixtures/lang")
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect();
    paths.sort();
    paths
        .into_iter()
        .map(|path| {
            let name = path.file_stem().unwrap().to_string_lossy().into_owned();
            let text = std::fs::read_to_string(&path).unwrap();
            let lines = text
                .lines()
                .map(|line| NFD.normalize(line, 0).unwrap().into_owned())
                .collect();
            (name, lines)
        })
        .collect()
}

/// Every Unicode scalar value once, in code point order, cut into lines.
fn all_code_points() -> Vec<String> {
    let chars: Vec<char> = (0..=char::MAX as u32).filter_map(char::from_u32).collect();
    chars
        .repeat(128)
        .chunks(SYNTHETIC_LINE_CHARS)
        .map(|line| line.iter().collect())
        .collect()
}

pub fn strip_accents(c: &mut Criterion) {
    let normalizer = StripAccents::new();
    let corpora = languages()
        .into_iter()
        .chain([("all_code_points".to_string(), all_code_points())]);

    for (name, lines) in corpora {
        let bytes: usize = lines.iter().map(String::len).sum();
        let mut group = c.benchmark_group(format!("strip_accents/{name}"));
        group.sampling_mode(SamplingMode::Flat);
        group.sample_size(20);
        group.throughput(Throughput::Bytes(bytes as u64));
        group.bench_function(VARIANT, |bencher| {
            bencher.iter(|| {
                for line in &lines {
                    black_box(normalizer.normalize(black_box(line), 0).unwrap());
                }
            })
        });
        group.finish();
    }
}

criterion_group!(benches, strip_accents);
criterion_main!(benches);
