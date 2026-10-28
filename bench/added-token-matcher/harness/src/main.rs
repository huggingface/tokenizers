//! Added-token matcher benchmark, one thread. Built twice: `--features buckets` against main (the bucket matcher),
//! and without it against the aho-corasick branch. Rows are tab-separated, one per measurement:
//!
//!   matcher <secs> <sets.tsv> <round>   BUILD and MATCHER rows: the matcher alone on each model's added tokens
//!   e2e <secs> <sets.tsv> <round>       LOAD and E2E rows: the whole tokenizer, `encode` on 4 KB pieces
use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::BTreeMap;
use std::hint::black_box;
use std::sync::atomic::{AtomicIsize, Ordering};
use std::time::Instant;

use tokenizers::pipeline::EncodeOptions;

#[cfg(feature = "buckets")]
const ARM: &str = "buckets";
#[cfg(not(feature = "buckets"))]
const ARM: &str = "aho-corasick";

struct Counting;
static LIVE: AtomicIsize = AtomicIsize::new(0);
static PEAK: AtomicIsize = AtomicIsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let live = LIVE.fetch_add(layout.size() as isize, Ordering::Relaxed) + layout.size() as isize;
        PEAK.fetch_max(live, Ordering::Relaxed);
        // SAFETY: forwards the caller's layout to the system allocator unchanged.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size() as isize, Ordering::Relaxed);
        // SAFETY: `ptr` came from `alloc` above with this layout.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// Runs `build`, returning its value with the bytes it left allocated and the peak above the starting point.
fn measured<T>(build: impl FnOnce() -> T) -> (T, isize, isize, f64) {
    let before = LIVE.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);
    let t0 = Instant::now();
    let value = build();
    let secs = t0.elapsed().as_secs_f64();
    (value, LIVE.load(Ordering::Relaxed) - before, PEAK.load(Ordering::Relaxed) - before, secs)
}

#[cfg(feature = "buckets")]
mod arm {
    pub type Matcher = tk_encode::vocab::buckets::Buckets;
    pub fn build(tokens: Vec<(Vec<u8>, u32)>) -> Matcher {
        Matcher::from_tokens(tokens).unwrap()
    }
    pub fn find(m: &Matcher, bytes: &[u8]) -> Option<(u32, usize, usize)> {
        let (id, start, len) = m.match_bytes(bytes)?;
        Some((id, start as usize, start as usize + len as usize))
    }
}

#[cfg(not(feature = "buckets"))]
mod arm {
    pub type Matcher = tk_encode::vocab::added_token_matcher::AddedTokenMatcher;
    pub fn build(tokens: Vec<(Vec<u8>, u32)>) -> Matcher {
        Matcher::new(tokens).unwrap()
    }
    pub fn find(m: &Matcher, bytes: &[u8]) -> Option<(u32, usize, usize)> {
        let (id, range) = m.find(bytes)?;
        Some((id, range.start, range.end))
    }
}

fn fnv(hash: u64, value: u64) -> u64 {
    (hash ^ value).wrapping_mul(0x100_0000_01b3)
}

const FNV_START: u64 = 0xcbf2_9ce4_8422_2325;

/// Every match in `bytes`, restarting after each match like `AddedVocabulary::extract_next` does for a token
/// without flags: the match count and a hash of every `(start, end, id)`.
fn scan(m: &arm::Matcher, bytes: &[u8]) -> (u64, u64) {
    let (mut search, mut count, mut hash) = (0, 0, FNV_START);
    while let Some((id, start, end)) = arm::find(m, &bytes[search..]) {
        hash = fnv(fnv(fnv(hash, (search + start) as u64), (search + end) as u64), id as u64);
        count += 1;
        search += end;
    }
    (count, hash)
}

/// The corpus cut into pieces of at most 4 KB at line ends (or a char boundary), the size of a document or a
/// chat turn rather than one 20 MB string.
fn pieces(text: &str) -> Vec<&str> {
    let (mut out, mut rest) = (Vec::new(), text);
    while !rest.is_empty() {
        let mut end = rest.len().min(4096);
        while !rest.is_char_boundary(end) {
            end -= 1;
        }
        if end < rest.len() {
            end = rest[..end].rfind('\n').map_or(end, |i| i + 1);
        }
        out.push(&rest[..end]);
        rest = &rest[end..];
    }
    out
}

/// MB/s of `pass` over `bytes` input bytes, repeated for at least `secs`.
fn throughput(bytes: usize, secs: f64, mut pass: impl FnMut()) -> f64 {
    let (t0, mut passes) = (Instant::now(), 0u64);
    while passes == 0 || t0.elapsed().as_secs_f64() < secs {
        pass();
        passes += 1;
    }
    bytes as f64 * passes as f64 / t0.elapsed().as_secs_f64() / 1e6
}

/// Every added token of a tokenizer.json as `(content, id)`, the set the matcher is built from.
fn added_tokens(path: &str) -> Vec<(Vec<u8>, u32)> {
    let json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let tokens: Vec<(Vec<u8>, u32)> = json["added_tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| (t["content"].as_str().unwrap().as_bytes().to_vec(), t["id"].as_u64().unwrap() as u32))
        .collect();
    tokens
}

/// The cells of sets.tsv grouped by model: `model -> (tokenizer.json, [(corpus, path)])`.
fn cells(path: &str) -> BTreeMap<String, (String, Vec<(String, String)>)> {
    let mut out: BTreeMap<String, (String, Vec<(String, String)>)> = BTreeMap::new();
    for line in std::fs::read_to_string(path).unwrap().lines() {
        let f: Vec<&str> = line.split('\t').collect();
        let entry = out.entry(f[0].to_string()).or_insert_with(|| (f[1].to_string(), Vec::new()));
        entry.1.push((f[2].to_string(), f[3].to_string()));
    }
    out
}

fn matcher(secs: f64, sets: &str, round: &str) {
    for (model, (tokenizer, corpora)) in cells(sets) {
        let tokens = added_tokens(&tokenizer);
        let (m, live, peak, build) = measured(|| arm::build(tokens.clone()));
        println!("BUILD\t{round}\t{ARM}\t{model}\t{}\t{:.1}\t{live}\t{peak}", tokens.len(), build * 1e6);
        for (corpus, path) in corpora {
            let text = std::fs::read(path).unwrap();
            let (count, hash) = scan(&m, &text);
            let whole = throughput(text.len(), secs, || {
                black_box(scan(&m, black_box(&text)));
            });
            println!("MATCHER\t{round}\t{ARM}\t{model}\t{corpus}\twhole\t{whole:.1}\t{count}\t{hash:016x}");
            let text = String::from_utf8(text).unwrap();
            let docs = pieces(&text);
            let (count, hash) = docs.iter().map(|d| scan(&m, d.as_bytes())).fold((0, FNV_START), |(c, h), (dc, dh)| (c + dc, fnv(h, dh)));
            let short = throughput(text.len(), secs, || {
                for d in &docs {
                    black_box(scan(&m, black_box(d.as_bytes())));
                }
            });
            println!("MATCHER\t{round}\t{ARM}\t{model}\t{corpus}\t4k\t{short:.1}\t{count}\t{hash:016x}");
        }
    }
}

fn e2e(secs: f64, sets: &str, round: &str) {
    tokenizers::parallelism::set_num_threads(1);
    let options = EncodeOptions::no_specials();
    for (model, (path, corpora)) in cells(sets) {
        let json = tokenizers::canonicalize_file(&path).unwrap();
        let (tokenizer, live, _, load) = measured(|| tokenizers::from_json(&json).unwrap());
        println!("LOAD\t{round}\t{ARM}\t{model}\t{:.2}\t{live}", load * 1e3);
        for (corpus, path) in corpora {
            let text = std::fs::read_to_string(path).unwrap();
            let docs = pieces(&text);
            let mut hash = FNV_START;
            for d in &docs {
                for t in tokenizer.encode(*d, &options).wait().unwrap()[0].ids() {
                    hash = fnv(hash, t.id() as u64);
                }
            }
            let mbs = throughput(text.len(), secs, || {
                for d in &docs {
                    drop(black_box(tokenizer.encode(black_box(*d), &options).wait().unwrap()));
                }
            });
            println!("E2E\t{round}\t{ARM}\t{model}\t{corpus}\t{mbs:.1}\t{hash:016x}");
        }
    }
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let secs: f64 = a[1].parse().unwrap();
    match a[0].as_str() {
        "matcher" => matcher(secs, &a[2], &a[3]),
        "e2e" => e2e(secs, &a[2], &a[3]),
        other => panic!("unknown mode {other}: matcher or e2e"),
    }
}
