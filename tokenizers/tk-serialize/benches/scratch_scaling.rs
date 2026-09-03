//! Does `PipelineTokenizer::encode` scale with thread count?
//!
//! N threads calling `encode` on one shared tokenizer, one row per thread count. Criterion
//! times the whole barrier-synced batch and `Throughput::Elements(threads)` turns that into
//! encodes/s, so the rows read straight off as a scaling curve and `cargo bench` compares
//! them against the previous run for you.
//!
//! `ScratchPool` is what this is pointed at: every `encode` takes a scratch from it and gives
//! it back on drop, so a pool that serialises caps the whole crate however well the encode
//! path itself scales.
//!
//! Two prompts per thread count, because the `WordCache` inside each scratch makes them
//! different benchmarks rather than the same one twice. `distinct` is real text and mostly
//! misses the cache; `repeated` is one line over and over and mostly hits, which is the
//! regime the cache was added for. A pool change should move both rows; a cache change moves
//! the second one only.

use std::hint::black_box;
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

use criterion::{
    BenchmarkId, Criterion, SamplingMode, Throughput, criterion_group, criterion_main,
};
use tk_encode::pipeline::PipelineTokenizer;

/// Roughly the prompt size the sharded pool was measured at in #2365.
const PROMPT_BYTES: usize = 12_000;

/// `deepseek-v4` for the same reason `encode.rs` uses it: a real pre-tokenizer, a BPE model
/// and 1283 added tokens, so an encode here is a whole pipeline rather than a lookup.
const FIXTURE: &str = "../data/deepseek-v4.json";

/// Read a real config from `../data`. They are all still version `1.0`, so run the upgrade pass
/// first -- the reader only accepts canonical `2.0`.
fn load(path: &str) -> PipelineTokenizer {
    let canonical = tk_convert::canonicalize_file(path).unwrap();
    tk_serialize::from_json(&canonical).unwrap()
}

/// Powers of two up to what the machine actually has, so the curve ends at full width. The
/// contention this bench is for only shows up in the last rows: the single mutex the pool used
/// to be scaled fine to 8 threads and knee'd between 16 and 32.
fn thread_counts() -> Vec<usize> {
    let max = std::thread::available_parallelism().map_or(1, |n| n.get());
    let mut counts: Vec<usize> = std::iter::successors(Some(1usize), |n| Some(n * 2))
        .take_while(|n| *n < max)
        .collect();
    counts.push(max);
    counts
}

/// A prompt of about [`PROMPT_BYTES`], built from real text so the word shapes are real too.
///
/// `repeated` picks the regime: one line over and over is a near-100% `WordCache` hit and
/// collapses the measured encode cost to cache lookups, which both inflates throughput and
/// inflates the share of each encode spent in the pool. Worth measuring -- it is a regime real
/// callers hit -- but only as its own row, never as the only one.
fn prompt(text: &str, repeated: bool) -> String {
    let mut lines = text.lines().filter(|line| !line.trim().is_empty()).cycle();
    let first = lines.next().unwrap();
    let mut out = String::with_capacity(PROMPT_BYTES + 256);
    while out.len() < PROMPT_BYTES {
        out.push_str(if repeated {
            first
        } else {
            lines.next().unwrap()
        });
        out.push('\n');
    }
    out
}

/// `iters` encodes on each of `threads` threads, all released together, timed as one batch.
fn encode_on(
    tokenizer: &Arc<PipelineTokenizer>,
    prompt: &Arc<String>,
    threads: usize,
    iters: u64,
) -> Duration {
    let barrier = Arc::new(Barrier::new(threads + 1));
    let handles: Vec<_> = (0..threads)
        .map(|_| {
            let (tokenizer, prompt, barrier) = (
                Arc::clone(tokenizer),
                Arc::clone(prompt),
                Arc::clone(&barrier),
            );
            std::thread::spawn(move || {
                barrier.wait();
                for _ in 0..iters {
                    // encode() hands back a handle; wait() is what produces the ids. Dropping
                    // the handle unwaited would measure the call and not the tokenizer.
                    black_box(tokenizer.encode(prompt.as_str(), true).wait().unwrap());
                }
            })
        })
        .collect();

    barrier.wait();
    let start = Instant::now();
    for handle in handles {
        handle.join().unwrap();
    }
    start.elapsed()
}

pub fn scratch_scaling(c: &mut Criterion) {
    let tokenizer = Arc::new(load(FIXTURE));
    let text = std::fs::read_to_string("../data/big.txt").unwrap();
    let prompts = [
        ("distinct", Arc::new(prompt(&text, false))),
        ("repeated", Arc::new(prompt(&text, true))),
    ];
    let threads = thread_counts();

    // The pool grows to the concurrency it has seen, so warm it at full width. Unwarmed, the
    // one-thread row would be the only one paying to populate it and would flatter every row
    // after it.
    black_box(encode_on(
        &tokenizer,
        &prompts[0].1,
        *threads.last().unwrap(),
        1,
    ));

    for (name, prompt) in &prompts {
        let mut group = c.benchmark_group(format!("scratch-scaling/{name}"));
        group.sampling_mode(SamplingMode::Flat);
        group.sample_size(10);
        group.warm_up_time(Duration::from_secs(1));
        group.measurement_time(Duration::from_secs(2));
        for &count in &threads {
            // Elements, not Bytes: one element is one whole encode, so the reported rate is
            // encodes/s across all threads and the row-over-row ratio is the speedup.
            group.throughput(Throughput::Elements(count as u64));
            group.bench_function(BenchmarkId::from_parameter(count), |b| {
                b.iter_custom(|iters| encode_on(&tokenizer, prompt, count, iters))
            });
        }
        group.finish();
    }
}

criterion_group!(scratch_scaling_benches, scratch_scaling);
criterion_main!(scratch_scaling_benches);
