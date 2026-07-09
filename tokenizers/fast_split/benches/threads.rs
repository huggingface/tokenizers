//! Thread-scaling of the pretokenizer pipeline (classify + fsm). The document is partitioned at clean
//! newline boundaries — every pretokenizer here ends a token at `[\r\n]`, so no token crosses a seam,
//! which the correctness check proves (N-way partitioned spans == sequential spans). Threads are
//! spawned ONCE via `std::thread::scope` (no external dep) and each loops its chunk `iters` times.
//!
//! English is ~memory-bound (classify is near-free → limited scaling); CJK is compute-bound (3-byte
//! classify dominates → scales close to linear). Run: cargo bench --bench threads
use fast_split::classify::{Atoms, classify};
use fast_split::fsm::{Span, fsm_byte_level, fsm_cl100k, fsm_deepseek};
use std::hint::black_box;
use std::time::Instant;

type Fsm = fn(&[u8], &[u8], &mut [Span]) -> usize;

const DOCS: &[(&str, &str)] = &[
    ("English", "../data/big.txt"),
    ("Chinese", "benches/data/zh.txt"),
    ("Japanese", "../data/unigram_wagahaiwa_nekodearu.txt"),
];
const PRETOKS: &[(&str, Fsm)] = &[
    ("cl100k", fsm_cl100k),
    ("deepseek", fsm_deepseek),
    ("byte_level", fsm_byte_level),
];

/// Partition `[0,len)` into ≤`n` chunks. Each seam is placed right after the LAST `\n` of a whitespace
/// run, so the whole `\s*[\r\n]+` token (which cl100k/deepseek build, absorbing trailing newlines) stays
/// intact in the preceding chunk — byte-exact seams for those. `byte_level` still shows a few Δ because
/// it tokenizes a trailing ws run differently at chunk-EOF vs mid-stream (→ overlap-chunk in production).
fn partition_nl(text: &[u8], n: usize) -> Vec<(usize, usize)> {
    let len = text.len();
    if n <= 1 || len == 0 {
        return vec![(0, len)];
    }
    let is_ws = |c: u8| matches!(c, b'\t' | b'\n' | 0x0B | 0x0C | b'\r' | b' ');
    let mut bounds = Vec::with_capacity(n);
    let mut start = 0;
    for k in 1..n {
        if start >= len {
            break;
        }
        let mut b = (len * k / n).max(start + 1).min(len);
        while b < len && text[b] != b'\n' {
            b += 1; // find a newline at/after the target
        }
        if b >= len {
            break;
        }
        let (mut last_nl, mut e) = (b, b); // extend over the ws run, tracking its last newline
        while e < len && is_ws(text[e]) {
            if text[e] == b'\n' {
                last_nl = e;
            }
            e += 1;
        }
        let seam = last_nl + 1; // right after the last newline of the run
        if seam > start && seam < len {
            bounds.push((start, seam));
            start = seam;
        }
    }
    bounds.push((start, len));
    bounds
}

fn pipeline(fsm: Fsm, chunk: &[u8], tags: &mut Vec<u8>, out: &mut Vec<Span>) {
    tags.clear();
    tags.resize(chunk.len(), 0);
    classify::<Atoms>(chunk, tags);
    out.clear();
    out.resize(chunk.len() + 1, (0, 0)); // preallocated slice — the fsm writes into it, no push
    let k = fsm(chunk, tags, out);
    out.truncate(k);
}

/// How many tokens differ between the N-way partitioned pipeline (offsets rebased) and the sequential
/// pipeline over the whole doc — 0 means the seams are byte-exact.
fn seam_diff(fsm: Fsm, pb: &[u8], n: usize) -> usize {
    use std::collections::HashSet;
    let (mut tags, mut seq) = (Vec::new(), Vec::new());
    pipeline(fsm, pb, &mut tags, &mut seq);
    let mut par = Vec::new();
    for &(st, en) in &partition_nl(pb, n) {
        let mut out = Vec::new();
        pipeline(fsm, &pb[st..en], &mut tags, &mut out);
        par.extend(out.iter().map(|&(a, b)| (a + st as u32, b + st as u32)));
    }
    if par == seq {
        return 0;
    }
    let (ss, ps): (HashSet<Span>, HashSet<Span>) =
        (seq.iter().copied().collect(), par.iter().copied().collect());
    par.iter().filter(|t| !ss.contains(t)).count() + seq.iter().filter(|t| !ps.contains(t)).count()
}

fn main() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let cores = std::thread::available_parallelism().map(|x| x.get()).unwrap_or(8);
    let mut threads = vec![1usize, 2, 4, 6, 8];
    if !threads.contains(&cores) {
        threads.push(cores);
    }
    threads.retain(|&t| t <= cores);
    println!("{cores} logical cores\n");

    for (dlabel, rel) in DOCS {
        let raw = match std::fs::read_to_string(format!("{manifest}/{rel}")) {
            Ok(s) if !s.trim().is_empty() => s,
            _ => {
                println!("{dlabel} (skipped — {rel} missing)\n");
                continue;
            }
        };
        // ~16 MB working buffer (defeats caches; amortizes spawn), ending on a newline.
        let reps = (16_000_000 / raw.len().max(1)).max(1);
        let mut doc = raw.repeat(reps);
        if !doc.ends_with('\n') {
            doc.push('\n');
        }
        let pb = doc.as_bytes();
        let iters = (400_000_000 / pb.len()).clamp(3, 40) as u32;

        for (plabel, fsm) in PRETOKS {
            let d = seam_diff(*fsm, pb, cores);
            let seam = if d == 0 { "exact".to_string() } else { format!("Δ{d} tok") };
            println!(
                "{dlabel} · {plabel}  ({:.1} MB, {iters} iters/thread, seams {seam})",
                pb.len() as f64 / 1e6
            );
            let mut base = 0.0;
            for &nt in &threads {
                let bounds = partition_nl(pb, nt);
                let t = Instant::now();
                std::thread::scope(|s| {
                    for &(st, en) in &bounds {
                        s.spawn(move || {
                            let (mut tags, mut out) = (Vec::new(), Vec::new());
                            for _ in 0..iters {
                                pipeline(*fsm, &pb[st..en], &mut tags, &mut out);
                                black_box(out.len());
                            }
                        });
                    }
                });
                let mbps = pb.len() as f64 * iters as f64 / t.elapsed().as_secs_f64() / 1e6;
                if nt == 1 {
                    base = mbps;
                }
                println!(
                    "  {nt:2} thr: {mbps:8.0} MB/s   {:5.2}x   {:3.0}% linear",
                    mbps / base,
                    mbps / base / nt as f64 * 100.0
                );
            }
            println!();
        }
    }
    println!(
        "(MB/s = bytes×iters / wall; %linear = speedup / threads. seams: partitioned-vs-sequential token\n\
         diff — cl100k/deepseek are exact at newline seams; byte_level's few Δ need overlap-chunk to erase.)"
    );
}
