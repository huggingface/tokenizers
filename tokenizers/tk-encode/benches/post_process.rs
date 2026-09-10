//! A/B rig for two claims made about `Template::post_process`/`wrap_pair`:
//! 1. `wrap_pair`'s `#[inline(never)]` protects the single-sequence path from inlining bloat.
//! 2. `SPECIALS` being a `const bool` (monomorphized) beats a runtime `bool` argument.
//!
//! Not meant to stay in the tree -- run once per source variant with `--save-baseline`/
//! `--baseline` to compare, per the investigation in chore/no-const-bools.

use std::hint::black_box;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use tk_encode::pipeline::{PipelineToken, Template};

fn bert_single() -> Template {
    Template {
        prefix: Box::new([(101.into(), 0)]),
        suffix: Box::new([(102.into(), 0)]),
        ..Template::default()
    }
}

fn bert_pair() -> Template {
    Template {
        prefix: Box::new([(101.into(), 0)]),
        infix: Box::new([(102.into(), 0)]),
        suffix: Box::new([(102.into(), 1)]),
        b_type_id: Some(1),
        ..Template::default()
    }
}

fn tokens(n: usize) -> Vec<PipelineToken> {
    (0..n as u32).map(PipelineToken::from).collect()
}

fn post_process(c: &mut Criterion) {
    let mut group = c.benchmark_group("post_process");
    for &len in &[8usize, 64] {
        let s1 = tokens(len);

        let single = bert_single();
        group.bench_function(format!("single/specials=true/len={len}"), |b| {
            b.iter_batched(
                || s1.clone(),
                |s1| black_box(single.post_process(s1, None)),
                BatchSize::SmallInput,
            )
        });
        group.bench_function(format!("single/specials=false/len={len}"), |b| {
            b.iter_batched(
                || s1.clone(),
                |s1| black_box(single.post_process_no_specials(s1, None)),
                BatchSize::SmallInput,
            )
        });

        let pair = bert_pair();
        let s2 = tokens(len);
        group.bench_function(format!("pair/specials=true/len={len}"), |b| {
            b.iter_batched(
                || (s1.clone(), Some(s2.clone())),
                |(a, b2)| black_box(pair.post_process(a, b2)),
                BatchSize::SmallInput,
            )
        });
        group.bench_function(format!("pair/specials=false/len={len}"), |b| {
            b.iter_batched(
                || (s1.clone(), Some(s2.clone())),
                |(a, b2)| black_box(pair.post_process_no_specials(a, b2)),
                BatchSize::SmallInput,
            )
        });
    }
    group.finish();
}

criterion_group!(post_process_benches, post_process);
criterion_main!(post_process_benches);
