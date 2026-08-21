//! Baseline harness for BPE training: per-phase timing plus a digest of the result.
//!
//! The digest is the gate. Any change to the training hot path has to leave it identical --
//! a faster trainer that moves an id is not a faster trainer, it is a different tokenizer.
//!
//! ```text
//! cargo run --release --example train_bench -- [vocab_size] [corpus]
//! cargo run --release --features count-allocs --example train_bench -- [vocab_size] [corpus]
//! ```
//!
//! The allocation counter is behind `count-allocs` and **must not** be on when timing anything.
//! It wraps every allocation in two atomic increments, which taxes allocation-heavy code far more
//! than allocation-light code: on the 6.5 MB corpus it inflated a pre-optimisation run from 677 ms
//! to 1.25 s while barely touching an optimised one, i.e. it flatters every speedup measured with
//! it. Time with the default features; count with the feature on.

use std::time::Instant;

use tk_train::{BpeTrainer, Trainer};

#[cfg(feature = "count-allocs")]
mod counting {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicU64, Ordering};

    pub static ALLOCS: AtomicU64 = AtomicU64::new(0);
    pub static BYTES: AtomicU64 = AtomicU64::new(0);

    pub struct Counting;

    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, l: Layout) -> *mut u8 {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(l.size() as u64, Ordering::Relaxed);
            unsafe { System.alloc(l) }
        }
        unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
            unsafe { System.dealloc(p, l) }
        }
        unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(new as u64, Ordering::Relaxed);
            unsafe { System.realloc(p, l, new) }
        }
    }

    pub fn snapshot() -> (u64, u64) {
        (
            ALLOCS.load(Ordering::Relaxed),
            BYTES.load(Ordering::Relaxed),
        )
    }
}

#[cfg(feature = "count-allocs")]
#[global_allocator]
static ALLOCATOR: counting::Counting = counting::Counting;

#[cfg(not(feature = "count-allocs"))]
fn snapshot() -> (u64, u64) {
    (0, 0)
}
#[cfg(feature = "count-allocs")]
use counting::snapshot;

/// FNV-1a. Hand-rolled because `ahash`'s default seed is per-process random, and a digest that
/// changes between runs cannot gate anything.
fn fnv(bytes: &[u8], mut h: u64) -> u64 {
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn main() {
    let mut args = std::env::args().skip(1);
    let vocab_size: usize = args.next().map_or(8000, |a| a.parse().unwrap());
    let corpus = args.next().unwrap_or_else(|| "../data/big.txt".to_string());

    let text = std::fs::read_to_string(&corpus).expect("corpus");
    println!(
        "corpus {corpus}: {:.1} MB, vocab_size {vocab_size}",
        text.len() as f64 / 1e6
    );

    let mut trainer = BpeTrainer::builder()
        .show_progress(false)
        .vocab_size(vocab_size)
        .build();

    // Whitespace split, so the measurement is the trainer and not a pre-tokenizer.
    let t = Instant::now();
    trainer
        .feed(text.lines(), |line| {
            Ok(line.split_whitespace().map(str::to_owned).collect())
        })
        .expect("feed");
    let feed = t.elapsed();

    let (a0, b0) = snapshot();
    let t = Instant::now();
    let (vocab, merges, _specials) = trainer.train_vocab().expect("train");
    let train = t.elapsed();
    let (a1, b1) = snapshot();

    // Digest: vocab in id order, then merges in rank order. Both are what a `tokenizer.json` holds.
    let mut by_id: Vec<(&u32, &String)> = vocab.iter().map(|(t, i)| (i, t)).collect();
    by_id.sort();
    let mut h = 0xcbf2_9ce4_8422_2325;
    for (id, token) in &by_id {
        h = fnv(&id.to_le_bytes(), h);
        h = fnv(token.as_bytes(), h);
    }
    for (a, b) in &merges {
        h = fnv(a.as_bytes(), h);
        h = fnv(b.as_bytes(), h);
    }

    println!("feed   {:>8.2?}", feed);
    println!("train  {:>8.2?}", train);
    println!("total  {:>8.2?}", feed + train);
    if cfg!(feature = "count-allocs") {
        println!(
            "train allocations: {} ({:.1} MB)",
            a1 - a0,
            (b1 - b0) as f64 / 1e6
        );
    }
    println!("vocab {} merges {}", vocab.len(), merges.len());
    println!("DIGEST {h:016x}");
}
