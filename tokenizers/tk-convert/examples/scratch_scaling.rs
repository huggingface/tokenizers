//! Does `PipelineTokenizer::encode` scale with thread count, and if not, where?
//!
//! Measuring Dynamo's frontend end to end says throughput stops improving past 16
//! cores and then declines, while the frontend's own CPU busy share falls from
//! 90% to 23%. Idle cores plus falling throughput means something serialises. The
//! load generator has been excluded (it sits at 10% busy in the cells where the
//! frontend goes idle) and so has the allocator (glibc against jemalloc is worth
//! 1.16x, not the knee).
//!
//! That leaves three candidates: this crate, the way Dynamo drives it, or a
//! configuration parameter. This example tests the first one in isolation, with
//! no Dynamo, no HTTP and no allocator games -- just N threads calling `encode`
//! on a shared tokenizer. If throughput here knees at 16 threads, the limit is in
//! the crate. If it scales to 88, the crate is exonerated and the search moves.
//!
//! There is a specific suspect. `ScratchPool` was a `Mutex<Vec<EncodeScratch>>`
//! and every `encode` locks it twice: once to pop a scratch, once to push it back
//! when the guard drops. The source carries HF's own note --
//! "TODO @McPatate : The Mutex can create contention, to be replaced by a better
//! access pattern" -- so this measures whether that TODO is the wall we are
//! hitting.
//!
//!   cargo run --release --example scratch_scaling -- \
//!       --tokenizer ~/.cache/huggingface/hub/models--gpt2/snapshots/*/tokenizer.json \
//!       --threads 1,2,4,8,16,32,64,88 --secs 5 --chars 12000

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

use tk_encode::pipeline::PipelineTokenizer;

/// Build a prompt of roughly `chars` bytes drawn from `vocab` distinct words.
///
/// Vocabulary size is the whole point of this knob. The BPE model keeps a
/// `WordCache` inside `PipelineModelScratch`, so a prompt cycling through a
/// handful of words is a near-100% cache hit and the measured encode cost
/// collapses to cache lookups. That inflates absolute throughput and, worse,
/// inflates the share of each encode spent on the pool mutex -- which is exactly
/// the quantity this benchmark was written to measure. A first version of this
/// file used 15 repeating words and has to be treated as void for that reason.
///
/// `--vocab 15` reproduces that behaviour so the artefact can be measured rather
/// than argued about. Natural text is nearer tens of thousands of distinct forms.
fn make_prompt(chars: usize, vocab: usize) -> String {
    let words: Vec<String> = (0..vocab.max(1)).map(word_for).collect();
    // A fixed LCG rather than a cycle: cycling through the vocabulary in order
    // is its own unrealistic pattern, and a fixed seed keeps runs comparable.
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut s = String::with_capacity(chars + 16);
    while s.len() < chars {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        s.push_str(&words[(state >> 33) as usize % words.len()]);
        s.push(' ');
    }
    s
}

/// A distinct lowercase word per index, 3-8 letters, so the vocabulary is made of
/// plausible word-shaped strings rather than numbers.
fn word_for(mut i: usize) -> String {
    let mut w = String::with_capacity(8);
    for _ in 0..3 {
        w.push((b'a' + (i % 26) as u8) as char);
        i /= 26;
    }
    while i > 0 {
        w.push((b'a' + (i % 26) as u8) as char);
        i /= 26;
    }
    w
}

fn run(tok: &Arc<PipelineTokenizer>, prompt: &Arc<String>, threads: usize, secs: u64) -> f64 {
    let barrier = Arc::new(Barrier::new(threads + 1));
    let stop = Arc::new(AtomicBool::new(false));
    let done = Arc::new(AtomicU64::new(0));
    let mut handles = Vec::with_capacity(threads);

    for _ in 0..threads {
        let (tok, prompt) = (Arc::clone(tok), Arc::clone(prompt));
        let (barrier, stop, done) = (Arc::clone(&barrier), Arc::clone(&stop), Arc::clone(&done));
        handles.push(std::thread::spawn(move || {
            let mut local = 0u64;
            barrier.wait();
            while !stop.load(Ordering::Relaxed) {
                // encode() hands back a handle; wait() is what produces the ids.
                // Dropping the handle unwaited would measure the call and not the
                // tokenizer.
                let enc = tok.encode(prompt.as_str(), true).wait();
                debug_assert!(enc.is_ok());
                std::hint::black_box(&enc);
                local += 1;
            }
            done.fetch_add(local, Ordering::Relaxed);
        }));
    }

    barrier.wait();
    let t0 = Instant::now();
    std::thread::sleep(Duration::from_secs(secs));
    stop.store(true, Ordering::Relaxed);
    for h in handles {
        let _ = h.join();
    }
    done.load(Ordering::Relaxed) as f64 / t0.elapsed().as_secs_f64()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut path = String::new();
    let mut threads = vec![1, 2, 4, 8, 16, 32, 64, 88];
    let mut secs = 5u64;
    let mut chars = 12000usize;
    let mut vocab = 30000usize;
    let mut i = 1;
    while i + 1 < args.len() {
        match args[i].as_str() {
            "--tokenizer" => path = args[i + 1].clone(),
            "--threads" => {
                threads = args[i + 1]
                    .split(',')
                    .filter_map(|x| x.trim().parse().ok())
                    .collect()
            }
            "--secs" => secs = args[i + 1].parse().unwrap_or(secs),
            "--chars" => chars = args[i + 1].parse().unwrap_or(chars),
            "--vocab" => vocab = args[i + 1].parse().unwrap_or(vocab),
            _ => {}
        }
        i += 2;
    }
    if path.is_empty() {
        eprintln!("need --tokenizer <path to tokenizer.json>");
        std::process::exit(2);
    }

    // tk_serialize reads canonical configs only, so a tokenizer.json written by any released
    // version of the library has to go through tk_convert first. That pairing is why this example
    // lives in tk-convert: it is the crate that has both on hand.
    let pipeline = match tk_convert::canonicalize_file(&path)
        .map_err(|e| format!("{e}"))
        .and_then(|json| tk_serialize::from_json(&json).map_err(|e| format!("{e:?}")))
    {
        Ok(p) => Arc::new(p),
        Err(e) => {
            eprintln!("cannot load {path}: {e}");
            std::process::exit(2);
        }
    };
    let prompt = Arc::new(make_prompt(chars, vocab));

    // Warm the scratch pool so the first row is not paying to populate it. The
    // pool grows to the concurrency it has seen, so an unwarmed single-thread
    // baseline would flatter every row after it.
    std::hint::black_box(run(&pipeline, &prompt, 4, 1));

    println!(
        "# scratch_scaling arch={} prompt_chars={} vocab={} secs={} tokens/encode~{}",
        std::env::consts::ARCH,
        prompt.len(),
        vocab,
        secs,
        pipeline
            .encode(prompt.as_str(), true)
            .wait()
            .map(|v| v.iter().map(|s| s.len()).sum::<usize>())
            .unwrap_or(0)
    );
    println!("threads\tencodes/s\tspeedup\tefficiency");
    let mut base = 0.0f64;
    for (n, &t) in threads.iter().enumerate() {
        let r = run(&pipeline, &prompt, t, secs);
        if n == 0 {
            base = r;
        }
        println!(
            "{}\t{:.1}\t{:.2}x\t{:.1}%",
            t,
            r,
            r / base,
            100.0 * (r / base) / t as f64
        );
    }
}
