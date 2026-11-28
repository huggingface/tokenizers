//! The optimized Unigram path must not allocate per encoded word once its scratch is warm.
//! Kept in its own test binary because it installs a counting global allocator.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use tk_encode::models::unigram::Unigram;
use tk_encode::tokenizer::pipeline::{Token, Tokenize};

struct Counting;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

fn model() -> Unigram {
    let vocab = vec![
        ("<unk>".to_string(), 0.0),
        ("a".to_string(), 1.0),
        ("b".to_string(), 1.0),
        ("aa".to_string(), 2.0),
        ("bb".to_string(), 2.0),
        ("ab".to_string(), 2.0),
        ("ba".to_string(), 2.0),
    ];
    Unigram::from(vocab, Some(0), false).unwrap()
}

#[test]
fn a_warm_scratch_encodes_without_allocating() {
    let model = model();
    let mut scratch = Tokenize::init_scratch(&model);
    let mut output = Vec::with_capacity(32);

    // Warm every buffer: the Viterbi table, the path spans, the id buffer, the word
    // cache and `output`. Same byte length so the table is not resized below.
    model.tokenize("abab", &mut scratch, &mut output).unwrap();
    assert!(!output.is_empty());
    output.clear();

    let before = ALLOCATIONS.load(Ordering::Relaxed);
    model.tokenize("baba", &mut scratch, &mut output).unwrap();
    let after = ALLOCATIONS.load(Ordering::Relaxed);

    let _: &Vec<Token> = &output;
    assert_eq!(after - before, 0, "the warm optimized path allocated");
}
