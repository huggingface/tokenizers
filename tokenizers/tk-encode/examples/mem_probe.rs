//! Live-heap probe: how much does a built pipeline hold? Allocation, not RSS —
//! building goes through a source `Tokenizer` whose freed pages never return to
//! the OS, so an RSS delta charges the pipeline for a structure it dropped.
//!
//! Usage: mem_probe <tokenizer.json>
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

static LIVE: AtomicUsize = AtomicUsize::new(0);

struct Counting;
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            LIVE.fetch_add(l.size(), Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        LIVE.fetch_sub(l.size(), Relaxed);
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        let q = unsafe { System.realloc(p, l, new) };
        if !q.is_null() {
            LIVE.fetch_add(new, Relaxed);
            LIVE.fetch_sub(l.size(), Relaxed);
        }
        q
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

fn main() {
    let model = std::env::args().nth(1).expect("tokenizer.json");
    let live0 = LIVE.load(Relaxed);
    let tok = tk_encode::Tokenizer::from_file(&model).expect("load");
    let pipeline =
        tk_encode::pipeline::PipelineTokenizer::try_from(&tok).expect("pipeline");
    drop(tok);
    let load = LIVE.load(Relaxed) - live0;
    let text = "The quick brown fox jumps over the lazy dog. ".repeat(2000);
    let n = pipeline.encode(&text, true).expect("encode").len();
    println!(
        "load {:.2} MB   after one encode {:.2} MB   ({n} ids)",
        load as f64 / 1048576.0,
        (LIVE.load(Relaxed) - live0) as f64 / 1048576.0
    );
}
