//! Count heap allocations on the pipeline encode hot path. Warms all
//! thread-local buffers first, resets the counter, then encodes one more pass
//! and reports allocs/dealloc/realloc — so a truly alloc-free hot path prints 0.
//! Run with POC_NOCACHE=1 to isolate the merge path (no cache).
//! Usage: xbench_alloc <tokenizer.json> <corpus(.json|.txt)>
use std::alloc::{GlobalAlloc, Layout, System};
use std::convert::TryFrom;
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use tk_encode::pipeline::PipelineTokenizer;
use tk_encode::Tokenizer;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static REALLOCS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);

struct Counting;
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(l.size(), Ordering::Relaxed);
        System.alloc(l)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l)
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        REALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(n, Ordering::Relaxed);
        System.realloc(p, l, n)
    }
}
#[global_allocator]
static A: Counting = Counting;

fn chunks_from(path: &str) -> Vec<String> {
    let raw = std::fs::read_to_string(path).unwrap();
    if path.ends_with(".json") {
        return serde_json::from_str(&raw).unwrap();
    }
    let mut out = Vec::new();
    let mut cur = String::new();
    for line in raw.lines() {
        cur.push_str(line);
        cur.push('\n');
        if cur.len() >= 10_000 {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let docs = chunks_from(&a[2]);
    let tok = Tokenizer::from_file(&a[1]).unwrap();
    let pipe = PipelineTokenizer::try_from(&tok).unwrap();
    let stage: u8 = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);
    let mut out = Vec::new();
    let mut pre = Vec::new();
    let mut run = || {
        let mut n = 0usize;
        for d in &docs {
            out.clear();
            pre.clear();
            match stage {
                0 => pipe.encode_generic::<0>(d, &mut out, &mut pre).unwrap(),
                1 => pipe.encode_generic::<1>(d, &mut out, &mut pre).unwrap(),
                2 => pipe.encode_generic::<2>(d, &mut out, &mut pre).unwrap(),
                _ => pipe.encode_generic::<3>(d, &mut out, &mut pre).unwrap(),
            }
            n += out.len();
        }
        n
    };
    // Warm: several passes so every reused buffer has grown to its max.
    for _ in 0..5 {
        black_box(run());
    }
    // Measured pass.
    let (a0, r0, b0) = (
        ALLOCS.load(Ordering::Relaxed),
        REALLOCS.load(Ordering::Relaxed),
        BYTES.load(Ordering::Relaxed),
    );
    black_box(run());
    let (a1, r1, b1) = (
        ALLOCS.load(Ordering::Relaxed),
        REALLOCS.load(Ordering::Relaxed),
        BYTES.load(Ordering::Relaxed),
    );
    println!(
        "docs={} allocs={} reallocs={} bytes={}  (per doc: {:.3} alloc, {:.3} realloc)",
        docs.len(),
        a1 - a0,
        r1 - r0,
        b1 - b0,
        (a1 - a0) as f64 / docs.len() as f64,
        (r1 - r0) as f64 / docs.len() as f64,
    );
}
