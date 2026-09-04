//! Repeated-batch throughput with equivalent worker-side token ID consumption.
//! Usage: hot_batch TOKENIZER_JSON CORPUS_TXT THREADS IDLE_SPIN_MICROS
use std::{
    cell::RefCell,
    hint::black_box,
    time::{Duration, Instant},
};
use tk_encode::{pipeline::PipelineTokenizer, utils::parallelism};

thread_local! {
    static IDS: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(
        args.len(),
        5,
        "usage: hot_batch TOKENIZER_JSON CORPUS_TXT THREADS IDLE_SPIN_MICROS"
    );
    let threads: usize = args[3].parse()?;
    let idle_us: u64 = args[4].parse()?;
    assert!(threads > 0);
    parallelism::set_num_threads(threads);
    parallelism::set_parallelism(true);
    parallelism::set_idle_spin_timeout(Duration::from_micros(idle_us));
    let canonical = tk_convert::canonicalize_file(&args[1])?;
    let pipeline: PipelineTokenizer = tk_serialize::from_json(&canonical)?;
    let corpus = std::fs::read_to_string(&args[2])?;
    let mut texts = Vec::new();
    let mut start = 0;
    while start < corpus.len() && texts.len() < 100 {
        let mut end = (start + 10 * 1024).min(corpus.len());
        while !corpus.is_char_boundary(end) {
            end += 1;
        }
        texts.push(&corpus[start..end]);
        start = end;
    }
    assert!(!texts.is_empty(), "empty corpus");
    let bytes: usize = texts.iter().map(|s| s.len()).sum();
    let once = || {
        pipeline.encode_batch_for_each(&texts, false, |_, encoding| {
            IDS.with(|ids| {
                let mut ids = ids.borrow_mut();
                ids.clear();
                ids.extend(encoding.ids().iter().map(|token| token.id()));
                black_box(&*ids);
            });
        })
    };
    for _ in 0..100 {
        once()?;
    }
    let mut seconds = Vec::new();
    for _ in 0..5 {
        let start = Instant::now();
        for _ in 0..50 {
            once()?;
        }
        seconds.push(start.elapsed().as_secs_f64());
    }
    // Check complete encodings, row order and exactly-once visitation outside timing.
    let expected = pipeline.encode(texts.as_slice(), false).wait()?;
    let seen = std::sync::Mutex::new(vec![None; texts.len()]);
    pipeline.encode_batch_for_each(&texts, false, |i, row| {
        assert!(seen.lock().unwrap()[i].replace(row.clone()).is_none());
    })?;
    let seen: Vec<_> = seen
        .into_inner()
        .unwrap()
        .into_iter()
        .map(Option::unwrap)
        .collect();
    assert_eq!(seen, expected);
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for token in expected.iter().flat_map(|row| row.ids()) {
        for byte in token.id().to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    let mut sorted = seconds.clone();
    sorted.sort_by(f64::total_cmp);
    println!(
        "{}",
        serde_json::json!({
            "threads": threads, "idle_spin_micros": idle_us, "corpus_bytes": bytes,
            "chunks": texts.len(), "warmup_batches": 100, "samples_seconds": seconds,
            "batches_per_sample": 50, "mib_s": bytes as f64 * 50.0 / 1048576.0 / sorted[2],
            "ids_hash": format!("{hash:016x}")
        })
    );
    Ok(())
}
