//! Per-operation timings for [`WordCache`], the table that remembers what a word
//! encoded to.
//!
//! ```text
//!   cargo run --release --features bench-internals --example word_cache_bench
//! ```
//!
//! # Comparing two versions of the cache
//!
//! Save a run, change the cache, run again against what you saved:
//!
//! ```text
//!   cargo run … --example word_cache_bench -- --save before.tsv
//!   # edit src/utils/word_cache.rs
//!   cargo run … --example word_cache_bench -- --baseline before.tsv
//! ```
//!
//! The second run prints a delta column. **Read it with the noise floor in
//! hand**, which the harness measures for you and prints at the top: one
//! scenario is run twice, from two call sites in the same binary, so the gap
//! between those two rows is the same work timed twice and nothing else.
//! Rebuilding moves code around, and on an M3 Max that alone has been worth 8%
//! to 13% on a single row. A delta smaller than the floor is not a result.
//!
//! If you need to resolve something smaller than that, this harness cannot do
//! it. Put both versions of the cache in one binary and alternate between them,
//! so neither gets to be the lucky one.
//!
//! # Reading the numbers
//!
//! Every row goes through the same loop, and the first row prices that loop on
//! its own — subtract it before quoting anything as the cost of an operation.
//!
//! The scenarios are named after the state the table is in, because that is what
//! decides the answer. A lookup spends most of its time waiting for memory, so
//! the same call costs two or three times as much against a table too big for
//! the last-level cache as against one that fits, and the big-table rows are
//! also the least repeatable — how the kernel maps those pages differs from one
//! process to the next. Give them `--reps 21` before believing a small
//! difference, or compare their `min` rather than their median.
//!
//! ```text
//!   --reps N           runs per scenario (default 7)
//!   --filter SUBSTR    only scenarios whose name contains SUBSTR
//!   --save PATH        write the results as TSV
//!   --baseline PATH    compare against a TSV written earlier
//! ```

use std::collections::HashMap;
use std::hint::black_box;
use std::time::Instant;
use tk_encode::utils::word_cache::{Lookup, WordCache};

/// Runs of one scenario. The median is the headline; the spread says whether to
/// believe it. Raise it with `--reps` for the memory-bound rows, which need more.
const DEFAULT_REPS: usize = 7;

/// Fixed-length words in one flat buffer, plus the scrambled order to walk them
/// in. One buffer rather than a `Vec` of `Vec`s so reading the next word costs
/// the measurement as little as possible, and scrambled so the hardware
/// prefetcher cannot do the table's work for it.
struct Pool {
    data: Vec<u8>,
    stride: usize,
    order: Vec<u32>,
}

impl Pool {
    /// `count` distinct words of exactly `stride` bytes. Two pools built with
    /// different `offset`s share no words, which is how the "this word is not in
    /// the table" scenarios get words that really are not in it.
    fn new(count: usize, stride: usize, offset: usize) -> Self {
        let mut data = Vec::with_capacity(count * stride);
        for i in 0..count {
            let word = format!("{n:0stride$}", n = i + offset);
            assert_eq!(word.len(), stride, "word {i} is not {stride} bytes");
            data.extend_from_slice(word.as_bytes());
        }
        let mut order: Vec<u32> = (0..count as u32).collect();
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        for i in (1..count).rev() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            order.swap(i, (state % (i as u64 + 1)) as usize);
        }
        Pool {
            data,
            stride,
            order,
        }
    }

    fn word(&self, i: usize) -> &[u8] {
        &self.data[i * self.stride..(i + 1) * self.stride]
    }

    fn count(&self) -> usize {
        self.data.len() / self.stride
    }
}

/// Look a word up and store what it encoded to, the two calls a caller that
/// misses always makes together.
fn store(cache: &mut WordCache, word: &[u8], ids: impl ExactSizeIterator<Item = u32>) {
    let at = match cache.lookup(word) {
        Lookup::Miss(at) => at,
        Lookup::Hit(_) => None,
    };
    if let Some(at) = at {
        cache.insert(at, ids);
    }
}

fn hit<'c>(lookup: Lookup<'c, '_>) -> Option<&'c [u32]> {
    match lookup {
        Lookup::Hit(ids) => Some(ids),
        Lookup::Miss(_) => None,
    }
}

/// A table with every slot taken, which is the state that makes a lookup walk
/// its whole window and an insert evict somebody.
fn saturated(slots: usize, ids: usize) -> WordCache {
    let filler = Pool::new(slots * 8, 8, 0);
    let mut cache = WordCache::new(slots);
    for i in 0..filler.count() {
        store(
            &mut cache,
            filler.word(i),
            (0..ids as u32).map(|k| k + i as u32),
        );
    }
    assert_eq!(
        cache.bench_occupancy(),
        slots,
        "the table is not saturated, so the walks below would stop early"
    );
    cache
}

struct Row {
    name: String,
    min: f64,
    median: f64,
    max: f64,
}

impl Row {
    fn new(name: String, mut ns: Vec<f64>) -> Self {
        ns.sort_by(f64::total_cmp);
        Row {
            name,
            min: ns[0],
            median: ns[ns.len() / 2],
            max: ns[ns.len() - 1],
        }
    }
}

/// Collects rows and prints them, against a saved run when there is one.
struct Bench {
    rows: Vec<Row>,
    reps: usize,
    filter: Option<String>,
    baseline: HashMap<String, (f64, f64, f64)>,
}

impl Bench {
    fn wanted(&self, name: &str) -> bool {
        self.filter
            .as_ref()
            .is_none_or(|needle| name.contains(needle.as_str()))
    }

    /// Time `op(iters)` `self.reps` times and keep the ns per operation. `op`
    /// returns a value that is fed to `black_box`, so the compiler cannot delete
    /// the work.
    fn run(&mut self, name: impl Into<String>, iters: usize, mut op: impl FnMut(usize) -> u64) {
        let name = name.into();
        if !self.wanted(&name) {
            return;
        }
        let mut ns = Vec::with_capacity(self.reps);
        for _ in 0..self.reps {
            let start = Instant::now();
            let sink = op(iters);
            let elapsed = start.elapsed();
            black_box(sink);
            ns.push(elapsed.as_nanos() as f64 / iters as f64);
        }
        self.rows.push(Row::new(name, ns));
    }

    fn print(&self) {
        let width = self.rows.iter().map(|r| r.name.len()).max().unwrap_or(0);
        println!(
            "\n  {:<width$} {:>8} {:>8} {:>8}{}",
            "",
            "min",
            "median",
            "max",
            if self.baseline.is_empty() {
                String::new()
            } else {
                format!("{:>10}", "vs base")
            }
        );
        for row in &self.rows {
            let delta = self.baseline.get(&row.name).map(|&(min, median, max)| {
                // Two runs are only telling you something different if their
                // spreads do not overlap. A median that has moved but still sits
                // inside the old run's range has not moved.
                let apart = row.min > max || row.max < min;
                let pct = (row.median - median) / median * 100.0;
                format!("{:>9.1}%{}", pct, if apart { " *" } else { "" })
            });
            println!(
                "  {:<width$} {:>8.2} {:>8.2} {:>8.2}{}",
                row.name,
                row.min,
                row.median,
                row.max,
                delta.unwrap_or_default()
            );
        }
        if !self.baseline.is_empty() {
            println!(
                "\n  * = the two runs' min..max ranges do not overlap. Everything else \
                 is inside\n      the spread of one run, whatever the median says."
            );
        }
    }

    fn tsv(&self) -> String {
        let mut out = String::new();
        for row in &self.rows {
            out.push_str(&format!(
                "{}\t{:.4}\t{:.4}\t{:.4}\n",
                row.name, row.min, row.median, row.max
            ));
        }
        out
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let save = flag("--save");
    let baseline_path = flag("--baseline");
    let baseline = baseline_path
        .as_ref()
        .map(|path| {
            std::fs::read_to_string(path)
                .unwrap_or_else(|e| panic!("cannot read baseline {path}: {e}"))
                .lines()
                .filter_map(|line| {
                    let mut fields = line.split('\t');
                    let name = fields.next()?.to_string();
                    let min = fields.next()?.parse().ok()?;
                    let median = fields.next()?.parse().ok()?;
                    let max = fields.next()?.parse().ok()?;
                    Some((name, (min, median, max)))
                })
                .collect()
        })
        .unwrap_or_default();

    let mut bench = Bench {
        rows: Vec::new(),
        reps: flag("--reps").map_or(DEFAULT_REPS, |n| n.parse().expect("--reps wants a number")),
        filter: flag("--filter"),
        baseline,
    };

    // Price the loop before anything else: this is what every row below carries.
    {
        let pool = Pool::new(1 << 14, 8, 0);
        let mut next = 0;
        bench.run("the loop, no cache call", 5_000_000, |iters| {
            let mut sink = 0;
            for _ in 0..iters {
                let word = pool.word(pool.order[next] as usize);
                next = if next + 1 == pool.count() {
                    0
                } else {
                    next + 1
                };
                sink += black_box(word)[0] as u64;
            }
            sink
        });
    }
    {
        let pool = Pool::new(1 << 14, 8, 0);
        let cache = WordCache::new(1 << 16);
        let mut next = 0;
        bench.run("+ pack the word and hash it", 5_000_000, |iters| {
            let mut sink = 0;
            for _ in 0..iters {
                let word = pool.word(pool.order[next] as usize);
                next = if next + 1 == pool.count() {
                    0
                } else {
                    next + 1
                };
                sink += black_box(&cache).bench_slot_key(black_box(word));
            }
            sink
        });
    }

    // Hits, over tables that fit in successively slower levels of the memory
    // hierarchy. Same code every time — the spread is all memory.
    for &(slots, label) in &[
        (1usize << 12, "hit, 2 ids in the slot   (4k slots, 128 KB)"),
        (1 << 16, "hit, 2 ids in the slot   (64k slots, 2 MB)"),
        // The same scenario a second time, from a second call site, so the gap
        // between the two rows prices this build's code layout. See the floor.
        (1 << 16, "  the row above again (control twin)"),
        (1 << 20, "hit, 2 ids in the slot   (1M slots, 32 MB)"),
    ] {
        let live = slots * 6 / 10;
        let pool = Pool::new(live, 8, 0);
        let mut cache = WordCache::new(slots);
        for i in 0..live {
            store(&mut cache, pool.word(i), [i as u32, 7].into_iter());
        }
        let mut next = 0;
        bench.run(label, 500_000, |iters| {
            let mut sink = 0;
            for _ in 0..iters {
                let word = pool.word(pool.order[next] as usize);
                next = if next + 1 == live { 0 } else { next + 1 };
                if let Some(ids) = hit(cache.lookup(black_box(word))) {
                    sink += ids[0] as u64;
                }
            }
            sink
        });
    }
    // Hits again, on the shapes that cost more than the common one: ids in the
    // arena, and a word too long to live in its own key.
    for &(stride, ids, label) in &[
        (8usize, 8usize, "hit, 8 ids in the arena  (64k slots)"),
        (24, 2, "hit, 24-byte word, hashed key  (64k slots)"),
        (24, 8, "hit, 24-byte word + 8 ids in the arena"),
    ] {
        let slots = 1 << 16;
        let live = slots * 6 / 10;
        let pool = Pool::new(live, stride, 0);
        let mut cache = WordCache::new(slots);
        for i in 0..live {
            store(
                &mut cache,
                pool.word(i),
                (0..ids as u32).map(|k| k + i as u32),
            );
        }
        let mut next = 0;
        bench.run(label, 500_000, |iters| {
            let mut sink = 0;
            for _ in 0..iters {
                let word = pool.word(pool.order[next] as usize);
                next = if next + 1 == live { 0 } else { next + 1 };
                if let Some(found) = hit(cache.lookup(black_box(word))) {
                    sink += found[0] as u64;
                }
            }
            sink
        });
    }

    // Misses. Nothing is stored, so the walk is the whole cost — and how far it
    // walks is the difference between these two rows.
    {
        let slots = 1 << 16;
        let live = slots / 4;
        let resident = Pool::new(live, 8, 0);
        let absent = Pool::new(1 << 16, 8, 10_000_000);
        let mut cache = WordCache::new(slots);
        for i in 0..live {
            store(&mut cache, resident.word(i), [i as u32, 7].into_iter());
        }
        let mut next = 0;
        bench.run(
            "miss, walk stops at an empty slot  (25% full)",
            500_000,
            |iters| {
                let mut sink = 0;
                for _ in 0..iters {
                    let word = absent.word(next);
                    next = if next + 1 == absent.count() {
                        0
                    } else {
                        next + 1
                    };
                    sink += u64::from(hit(cache.lookup(black_box(word))).is_none());
                }
                sink
            },
        );
    }
    {
        let mut cache = saturated(1 << 12, 2);
        let absent = Pool::new(1 << 16, 8, 10_000_000);
        let mut next = 0;
        bench.run(
            "miss, all 16 walked and scored  (saturated)",
            500_000,
            |iters| {
                let mut sink = 0;
                for _ in 0..iters {
                    let word = absent.word(next);
                    next = if next + 1 == absent.count() {
                        0
                    } else {
                        next + 1
                    };
                    sink += u64::from(hit(cache.lookup(black_box(word))).is_none());
                }
                sink
            },
        );
    }

    // What a word costs the first time it is ever seen: the miss above, plus
    // storing what the model worked out. Subtract the matching miss row to see
    // what the insert itself adds.
    {
        let iters = 60_000;
        let pool = Pool::new(iters, 8, 0);
        let mut ns = Vec::with_capacity(bench.reps);
        let name = "miss + insert, free slot, 2 ids  (46% full)";
        if bench.wanted(name) {
            for _ in 0..bench.reps {
                // Building the table is not part of the measurement.
                let mut cache = WordCache::new(1 << 17);
                let start = Instant::now();
                for i in 0..iters {
                    store(&mut cache, pool.word(i), [i as u32, 7].into_iter());
                }
                ns.push(start.elapsed().as_nanos() as f64 / iters as f64);
                black_box(&cache);
            }
            bench.rows.push(Row::new(name.to_string(), ns));
        }
    }
    for &(stride, ids, label) in &[
        (
            8usize,
            2usize,
            "miss + insert, evicting, 2 ids  (saturated)",
        ),
        (8, 8, "miss + insert, evicting, 8 ids to the arena"),
        (24, 8, "miss + insert, evicting, 24-byte word + 8 ids"),
    ] {
        let mut cache = saturated(1 << 12, ids);
        let fresh = Pool::new(1 << 17, stride, 10_000_000);
        let mut next = 0;
        bench.run(label, 200_000, |iters| {
            for _ in 0..iters {
                let word = fresh.word(next);
                next = if next + 1 == fresh.count() {
                    0
                } else {
                    next + 1
                };
                store(
                    &mut cache,
                    black_box(word),
                    (0..ids as u32).map(|k| k + next as u32),
                );
            }
            0
        });
        // An arena with nothing left in it turns inserts away, and a run of
        // refusals would time as something far cheaper than an eviction.
        let last = if next == 0 {
            fresh.count() - 1
        } else {
            next - 1
        };
        assert!(
            hit(cache.lookup(fresh.word(last))).is_some(),
            "{label}: the last insert did not land — the arena ran out"
        );
    }

    // Keeping the counters honest. Neither of these is on the path of a lookup;
    // the question is only whether they are cheap enough to ignore.
    {
        let mut cache = saturated(1 << 12, 2);
        bench.run(
            "fade(), the bumps that do not start over",
            5_000_000,
            |iters| {
                for _ in 0..iters {
                    black_box(&mut cache).bench_fade();
                }
                0
            },
        );
    }
    {
        let slots = 1 << 16;
        let live = slots * 6 / 10;
        let pool = Pool::new(live, 8, 0);
        let mut cache = WordCache::new(slots);
        for i in 0..live {
            store(&mut cache, pool.word(i), [i as u32, 7].into_iter());
        }
        bench.run(
            "restart_epochs(), one whole-table pass  (64k slots)",
            200,
            |iters| {
                for _ in 0..iters {
                    black_box(&mut cache).bench_restart_epochs();
                }
                0
            },
        );
    }

    // The floor: identical work at two call sites in this same binary. Anything
    // smaller than this is telling you about code layout, not about the cache.
    let timed = |name: &str| {
        bench
            .rows
            .iter()
            .find(|row| row.name == name)
            .map(|row| row.median)
    };
    // Nothing to price the build with if --filter dropped one of the pair.
    if let (Some(once), Some(twice)) = (
        timed("hit, 2 ids in the slot   (64k slots, 2 MB)"),
        timed("  the row above again (control twin)"),
    ) {
        let pct = (twice - once).abs() / once * 100.0;
        println!(
            "\nnoise floor for this build: {pct:.1}% (one scenario, two call sites, one binary)"
        );
    }
    bench.print();

    if let Some(path) = save {
        std::fs::write(&path, bench.tsv()).unwrap_or_else(|e| panic!("cannot write {path}: {e}"));
        println!("\nsaved to {path} — rerun with --baseline {path} after changing the cache");
    }
    println!();
}
