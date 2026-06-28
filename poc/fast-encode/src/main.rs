// POC: zero-alloc, range-based llama-3 encode + IREE-style ring buffer + ILP-tuned BPE.
// - encode_lvl<const L>: ablation. L=1 special-match, 2 +regex split, 3 +byte map, 4 +vocab lookup,
//   5 +BPE merge (full). Diffing the levels gives the per-stage cost with no per-call timer overhead.
// - encode_ring: process the input through a fixed power-of-2 ring (IREE transform-buffer style):
//   bounded memory, cache-resident working set, complete-token boundary carry (parity preserved).
// - merge pair-search uses two interleaved accumulators for ILP.
use ahash::AHashMap;
use ahash::RandomState;
use memchr::memchr;
use ptr_hash::bucket_fn::Linear;
use ptr_hash::{PtrHash, PtrHashParams};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use onig::Regex;
use serde_json::Value;
use std::hint::black_box;
use std::time::Instant;
mod dfa;
use dfa::{build_cls, first_not, Cls};

// counts allocation CALLS (not bytes) so we can prove the hot path is allocation-free.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
static ALLOCS: AtomicU64 = AtomicU64::new(0);
struct Counting;
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 { ALLOCS.fetch_add(1, Relaxed); System.alloc(l) }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) { System.dealloc(p, l) }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 { ALLOCS.fetch_add(1, Relaxed); System.realloc(p, l, n) }
}
#[global_allocator] static GA: Counting = Counting;

// Minimal MPHF VocabStore (ptr_hash): hash key -> slot -> one slab byte-compare. Strings stored once.
type Mphf = PtrHash<u64, Linear>;
#[derive(Clone, Copy)]
struct VEntry { start: u32, len: u32, id: u32 }
struct VocabStore { mphf: Mphf, hasher: RandomState, bytes: Box<[u8]>, entries: Box<[VEntry]> }
impl VocabStore {
    fn build(tokens: &[(Vec<u8>, u32)]) -> Self {
        let hasher = RandomState::with_seeds(0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344);
        let keys: Vec<u64> = tokens.iter().map(|(s, _)| hasher.hash_one(s.as_slice())).collect();
        let mut params = PtrHashParams::default_fast();
        params.single_part = true;
        let mphf = Mphf::new(&keys, params);
        let n = tokens.len();
        let mut bytes = Vec::with_capacity(tokens.iter().map(|(s, _)| s.len()).sum());
        let mut entries = vec![VEntry { start: 0, len: 0, id: 0 }; n];
        for (s, id) in tokens {
            let slot = mphf.index_single_part(&hasher.hash_one(s.as_slice()));
            entries[slot] = VEntry { start: bytes.len() as u32, len: s.len() as u32, id: *id };
            bytes.extend_from_slice(s);
        }
        VocabStore { mphf, hasher, bytes: bytes.into_boxed_slice(), entries: entries.into_boxed_slice() }
    }
    #[inline]
    fn get(&self, q: &[u8]) -> Option<u32> {
        let slot = self.mphf.index_single_part(&self.hasher.hash_one(q));
        let e = self.entries[slot];
        let (s, l) = (e.start as usize, e.len as usize);
        if l == q.len() && self.bytes[s..s + l] == *q { Some(e.id) } else { None }
    }
}

// Thread-local pretoken->ids cache. Open-addressing slots + two bump arenas, clear-on-full.
// Owned per encode thread (passed &mut) => NO lock, NO contention (fixes HF main's shared-lock design).
// Allocation-free in steady state: arenas pre-reserved; clear() resets lengths, never frees.
#[derive(Clone, Copy)]
struct CSlot { hash: u64, koff: u32, ioff: u32, klen: u16, ilen: u16 }
struct FlatCache { slots: Box<[CSlot]>, kbytes: Vec<u8>, ids: Vec<u32>, mask: usize, count: usize, hasher: RandomState }
impl FlatCache {
    fn new() -> Self {
        let bits = std::env::var("CACHE_BITS").ok().and_then(|s| s.parse().ok()).unwrap_or(16usize);
        Self::with_bits(bits)
    }
    fn with_bits(bits: usize) -> Self {
        let slots_n = 1usize << bits;
        let slots = vec![CSlot { hash: 0, koff: 0, ioff: 0, klen: 0, ilen: 0 }; slots_n].into_boxed_slice();
        FlatCache {
            slots, mask: slots_n - 1, count: 0,
            kbytes: Vec::with_capacity(slots_n * 16), // ~16 B/pretoken
            ids: Vec::with_capacity(slots_n * 8),
            hasher: RandomState::with_seeds(0xC0FFEE, 0xBADF00D, 0xDEADBEEF, 0x1337),
        }
    }
    #[inline]
    fn clear(&mut self) { for s in self.slots.iter_mut() { s.klen = 0; } self.kbytes.clear(); self.ids.clear(); self.count = 0; }
    #[inline]
    fn get(&self, p: &[u8], h: u64) -> Option<(u32, u16)> {
        let mut i = (h as usize) & self.mask;
        loop {
            let s = self.slots[i];
            if s.klen == 0 { return None; } // empty slot (pretokens are never empty)
            if s.hash == h && s.klen as usize == p.len()
                && self.kbytes[s.koff as usize..s.koff as usize + s.klen as usize] == *p { return Some((s.ioff, s.ilen)); }
            i = (i + 1) & self.mask;
        }
    }
    #[inline]
    fn insert(&mut self, p: &[u8], h: u64, ids: &[u32]) {
        if p.len() > u16::MAX as usize || ids.len() > u16::MAX as usize { return; }
        if self.kbytes.len() + p.len() > self.kbytes.capacity()
            || self.ids.len() + ids.len() > self.ids.capacity()
            || self.count * 4 >= self.slots.len() * 3 { self.clear(); }
        let (koff, ioff) = (self.kbytes.len() as u32, self.ids.len() as u32);
        self.kbytes.extend_from_slice(p);
        self.ids.extend_from_slice(ids);
        let mut i = (h as usize) & self.mask;
        while self.slots[i].klen != 0 { i = (i + 1) & self.mask; }
        self.slots[i] = CSlot { hash: h, koff, ioff, klen: p.len() as u16, ilen: ids.len() as u16 };
        self.count += 1;
    }
}

const PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

thread_local! { static RE: onig::Regex = onig::Regex::new(PATTERN).unwrap(); }

fn bytes_to_unicode() -> [char; 256] {
    let mut bs: Vec<u32> = Vec::new();
    for b in 0x21..=0x7e { bs.push(b); }
    for b in 0xa1..=0xac { bs.push(b); }
    for b in 0xae..=0xff { bs.push(b); }
    let inbs: std::collections::HashSet<u32> = bs.iter().copied().collect();
    let (mut bs2, mut cs2) = (bs.clone(), bs.clone());
    let mut n = 0u32;
    for b in 0u32..256 { if !inbs.contains(&b) { bs2.push(b); cs2.push(256 + n); n += 1; } }
    let mut out = ['\0'; 256];
    for i in 0..bs2.len() { out[bs2[i] as usize] = char::from_u32(cs2[i]).unwrap(); }
    out
}

struct Bpe {
    vocab: AHashMap<Vec<u8>, u32>,
    vocab_raw: AHashMap<Vec<u8>, u32>, // keyed by RAW bytes (byte-level inverted) -> no map expansion
    vs: Option<VocabStore>,            // same raw-byte vocab as an MPHF (strings once, 5.8x less RAM)
    ranks: AHashMap<(u32, u32), (u32, u32)>,
    byte_str: Vec<Vec<u8>>,
    byte_id: [u32; 256],
}
// reusable scratch for the heap+linked-list merge
struct MergeScratch { next: Vec<i32>, prev: Vec<i32>, alive: Vec<bool>, heap: BinaryHeap<Reverse<(u32, u32)>> }
impl MergeScratch {
    // pre-allocated for worst-case pretoken length; the hot path never allocates after this.
    fn new() -> Self { Self::with_capacity(1 << 14) } // 16384 symbols
    fn with_capacity(cap: usize) -> Self {
        MergeScratch { next: Vec::with_capacity(cap), prev: Vec::with_capacity(cap), alive: Vec::with_capacity(cap), heap: BinaryHeap::with_capacity(2 * cap) }
    }
    // grow once if a pathological pretoken exceeds the reservation (rare); keeps the push loop realloc-free.
    #[inline]
    fn ensure(&mut self, n: usize) {
        if self.next.capacity() < n { self.next.reserve(n); self.prev.reserve(n); self.alive.reserve(n); self.heap.reserve(2 * n); }
    }
}

// Incremental re-encode cache (multi-turn chat: reuse the unchanged prefix's split + ids).
struct PrefixCache { prev: Vec<u8>, ids: Vec<u32>, ends: Vec<(u32, u32)> } // ends: (byte_end, id_count) per emitted unit
impl PrefixCache { fn new() -> Self { PrefixCache { prev: Vec::new(), ids: Vec::new(), ends: Vec::new() } } }
#[inline]
fn common_prefix_len(a: &[u8], b: &[u8]) -> usize {
    let n = a.len().min(b.len());
    let mut i = 0;
    while i + 8 <= n { if a[i..i + 8] != b[i..i + 8] { break; } i += 8; }
    while i < n && a[i] == b[i] { i += 1; }
    i
}
impl Bpe {
    fn load(path: &str) -> Self {
        let j: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let m = &j["model"];
        let mut vocab: AHashMap<Vec<u8>, u32> = AHashMap::new();
        for (k, v) in m["vocab"].as_object().unwrap() { vocab.insert(k.as_bytes().to_vec(), v.as_u64().unwrap() as u32); }
        let b2u = bytes_to_unicode();
        let byte_str: Vec<Vec<u8>> = (0..256).map(|b| b2u[b].to_string().into_bytes()).collect();
        let mut byte_id = [0u32; 256];
        for b in 0..256 { byte_id[b] = *vocab.get(&byte_str[b]).unwrap(); }
        let mut ranks = AHashMap::new();
        for (rank, pair) in m["merges"].as_array().unwrap().iter().enumerate() {
            let (a, b) = match pair {
                Value::Array(xs) => (xs[0].as_str().unwrap(), xs[1].as_str().unwrap()),
                Value::String(s) => { let mut it = s.splitn(2, ' '); (it.next().unwrap(), it.next().unwrap()) }
                _ => continue,
            };
            let (ka, kb) = (a.as_bytes().to_vec(), b.as_bytes().to_vec());
            let mut ab = ka.clone(); ab.extend_from_slice(&kb);
            if let (Some(&ia), Some(&ib), Some(&iab)) = (vocab.get(&ka), vocab.get(&kb), vocab.get(&ab)) {
                ranks.insert((ia, ib), (rank as u32, iab));
            }
        }
        // vocab keyed by RAW bytes: invert the byte-level bijection on each vocab key.
        let mut char2byte: AHashMap<char, u8> = AHashMap::new();
        for b in 0..256 { char2byte.insert(b2u[b], b as u8); }
        let mut vocab_raw: AHashMap<Vec<u8>, u32> = AHashMap::with_capacity(vocab.len());
        for (k, &id) in &vocab {
            let s = std::str::from_utf8(k).unwrap();
            let mut raw = Vec::with_capacity(s.len());
            let mut ok = true;
            for ch in s.chars() { if let Some(&b) = char2byte.get(&ch) { raw.push(b); } else { ok = false; break; } }
            if ok { vocab_raw.insert(raw, id); }
        }
        let vs = Some(VocabStore::build(&vocab_raw.iter().map(|(k, &v)| (k.clone(), v)).collect::<Vec<_>>()));
        Bpe { vocab, vocab_raw, vs, ranks, byte_str, byte_id }
    }
    // heap + linked-list BPE merge (O(n log n)); appends final ids to `out`.
    #[inline]
    fn merge_heap(&self, syms: &mut Vec<u32>, out: &mut Vec<u32>, ms: &mut MergeScratch) {
        let n = syms.len();
        ms.next.clear(); ms.prev.clear(); ms.alive.clear(); ms.heap.clear();
        ms.ensure(n); // realloc-free build (pre-sized; grows only for a pathological pretoken)
        for i in 0..n { ms.next.push(i as i32 + 1); ms.prev.push(i as i32 - 1); ms.alive.push(true); }
        for i in 0..n - 1 {
            if let Some(&(r, _)) = self.ranks.get(&(syms[i], syms[i + 1])) { ms.heap.push(Reverse((r, i as u32))); }
        }
        while let Some(Reverse((r, pos))) = ms.heap.pop() {
            let i = pos as usize;
            if !ms.alive[i] { continue; }
            let j = ms.next[i];
            if j < 0 || j as usize >= n || !ms.alive[j as usize] { continue; }
            let j = j as usize;
            match self.ranks.get(&(syms[i], syms[j])) {
                Some(&(rr, m)) if rr == r => {
                    syms[i] = m;
                    ms.alive[j] = false;
                    let nj = ms.next[j];
                    ms.next[i] = nj;
                    if nj >= 0 && (nj as usize) < n { ms.prev[nj as usize] = i as i32; }
                    let pi = ms.prev[i];
                    if pi >= 0 { if let Some(&(r2, _)) = self.ranks.get(&(syms[pi as usize], syms[i])) { ms.heap.push(Reverse((r2, pi as u32))); } }
                    if nj >= 0 && (nj as usize) < n { if let Some(&(r2, _)) = self.ranks.get(&(syms[i], syms[nj as usize])) { ms.heap.push(Reverse((r2, i as u32))); } }
                }
                _ => {}
            }
        }
        let mut i = 0usize;
        loop { out.push(syms[i]); let nx = ms.next[i]; if nx < 0 || (nx as usize) >= n { break; } i = nx as usize; }
    }
    // linear O(n^2) merge with ILP dual-accumulator; no heap/linked-list setup. Wins for SHORT pretokens.
    #[inline]
    fn merge_linear(&self, syms: &mut Vec<u32>, out: &mut Vec<u32>) {
        loop {
            let n = syms.len();
            if n < 2 { break; }
            let (mut r0, mut i0, mut m0) = (u32::MAX, usize::MAX, 0u32);
            let (mut r1, mut i1, mut m1) = (u32::MAX, usize::MAX, 0u32);
            let mut i = 0;
            while i + 1 < n {
                if let Some(&(r, m)) = self.ranks.get(&(syms[i], syms[i + 1])) { if r < r0 { r0 = r; i0 = i; m0 = m; } }
                if i + 2 < n { if let Some(&(r, m)) = self.ranks.get(&(syms[i + 1], syms[i + 2])) { if r < r1 { r1 = r; i1 = i + 1; m1 = m; } } }
                i += 2;
            }
            let (bi, bm) = if r0 < r1 || (r0 == r1 && i0 <= i1) { (i0, m0) } else { (i1, m1) };
            if bi == usize::MAX { break; }
            syms[bi] = bm; syms.remove(bi + 1);
        }
        out.extend_from_slice(syms);
    }
    // hybrid: linear for short pretokens (no heap overhead), heap for long ones (O(n log n)).
    const MERGE_HEAP_MIN: usize = 24;
    #[inline]
    fn merge(&self, syms: &mut Vec<u32>, out: &mut Vec<u32>, ms: &mut MergeScratch) {
        if syms.len() < Self::MERGE_HEAP_MIN { self.merge_linear(syms, out); } else { self.merge_heap(syms, out, ms); }
    }
    // optimized full piece: raw-byte vocab (no byte-level map) + hybrid merge.
    #[inline]
    fn piece5(&self, p: &[u8], out: &mut Vec<u32>, syms: &mut Vec<u32>, ms: &mut MergeScratch) {
        if p.is_empty() { return; }
        if let Some(&id) = self.vocab_raw.get(p) { out.push(id); return; }
        syms.clear();
        if syms.capacity() < p.len() { syms.reserve(p.len()); } // realloc-free symbol fill
        for &b in p { syms.push(self.byte_id[b as usize]); }
        if syms.len() == 1 { out.push(syms[0]); return; }
        self.merge(syms, out, ms);
    }
    // same as piece5 but bytes->id via the MPHF VocabStore instead of the AHashMap
    #[inline]
    fn piece5_mphf(&self, p: &[u8], out: &mut Vec<u32>, syms: &mut Vec<u32>, ms: &mut MergeScratch) {
        if p.is_empty() { return; }
        if let Some(id) = self.vs.as_ref().unwrap().get(p) { out.push(id); return; }
        syms.clear();
        if syms.capacity() < p.len() { syms.reserve(p.len()); } // realloc-free symbol fill
        for &b in p { syms.push(self.byte_id[b as usize]); }
        if syms.len() == 1 { out.push(syms[0]); return; }
        self.merge(syms, out, ms);
    }
    #[inline]
    fn piece<const L: u8>(&self, p: &[u8], out: &mut Vec<u32>, map: &mut Vec<u8>, syms: &mut Vec<u32>) {
        if p.is_empty() { return; }
        if L >= 3 { map.clear(); for &b in p { map.extend_from_slice(&self.byte_str[b as usize]); } }
        if L >= 4 { if let Some(&id) = self.vocab.get(map.as_slice()) { if L >= 5 { out.push(id); } return; } }
        if L >= 5 {
            syms.clear();
            for &b in p { syms.push(self.byte_id[b as usize]); }
            loop {
                let n = syms.len();
                if n < 2 { break; }
                // ILP: two interleaved min-accumulators over independent rank lookups
                let (mut r0, mut i0, mut m0) = (u32::MAX, usize::MAX, 0u32);
                let (mut r1, mut i1, mut m1) = (u32::MAX, usize::MAX, 0u32);
                let mut i = 0;
                while i + 1 < n {
                    if let Some(&(r, m)) = self.ranks.get(&(syms[i], syms[i + 1])) { if r < r0 { r0 = r; i0 = i; m0 = m; } }
                    if i + 2 < n { if let Some(&(r, m)) = self.ranks.get(&(syms[i + 1], syms[i + 2])) { if r < r1 { r1 = r; i1 = i + 1; m1 = m; } } }
                    i += 2;
                }
                // leftmost lowest-rank wins (BPE merges the earliest occurrence on ties)
                let (bi, bid) = if r0 < r1 || (r0 == r1 && i0 <= i1) { (i0, m0) } else { (i1, m1) };
                if bi == usize::MAX { break; }
                syms[bi] = bid;
                syms.remove(bi + 1);
            }
            out.extend_from_slice(syms);
        }
    }
}

struct Specials { set: AHashMap<Vec<u8>, u32>, lens: Vec<usize>, first: u8, single: bool, cand: [bool; 256] }
impl Specials {
    fn new(items: &[(Vec<u8>, u32)]) -> Self {
        let set: AHashMap<Vec<u8>, u32> = items.iter().cloned().collect();
        let mut lens: Vec<usize> = items.iter().map(|(s, _)| s.len()).collect();
        lens.sort_unstable_by(|a, b| b.cmp(a)); lens.dedup();
        let mut cand = [false; 256];
        for (s, _) in items { cand[s[0] as usize] = true; }
        Specials { set, lens, first: items[0].0[0], single: items.iter().all(|(s, _)| s[0] == items[0].0[0]), cand }
    }
    #[inline]
    fn next(&self, t: &[u8], from: usize) -> Option<(usize, usize, u32)> {
        let mut s = from;
        loop {
            let rel = if self.single { memchr(self.first, &t[s..])? } else { t[s..].iter().position(|&c| self.cand[c as usize])? };
            let ms = s + rel;
            let rem = &t[ms..];
            for &l in &self.lens { if l <= rem.len() { if let Some(&id) = self.set.get(&rem[..l]) { return Some((ms, l, id)); } } }
            s = ms + 1;
        }
    }
}

// Fast DFA pre-tokenizer for the ASCII common case: replicates the GPT-4 regex rules 1-4
// (contractions, words, 1-3 digits, punctuation runs, each with an optional single leading space/char).
// Returns Some(end) when it can decide the token; None to fall back to onig (non-ASCII byte involved,
// or a whitespace-run case handled by rules 5/6). Same ordering as the regex => same boundaries.
#[inline]
fn letter(c: u8) -> bool { c.is_ascii_alphabetic() }
#[inline]
fn digit(c: u8) -> bool { c.is_ascii_digit() }
#[inline]
fn nl(c: u8) -> bool { c == b'\n' || c == b'\r' }
#[inline]
fn wsa(c: u8) -> bool { c == b' ' || (0x09..=0x0d).contains(&c) }

// NEON-accelerated DFA pre-tokenizer. Run-ends found via the nibble classifier; pure-space runs
// (rules 5/6) handled in the fast path. Returns None to fall back to onig (non-ASCII / newline / gap).
#[inline]
fn fast_token(t: &[u8], i: usize, end: usize, cl_l: &Cls, cl_p: &Cls, cl_s: &Cls) -> Option<usize> {
    let b = t[i];
    if b >= 0x80 { return None; }
    // Rule 1: (?i:'s|'t|'re|'ve|'m|'ll|'d)
    if b == b'\'' && i + 1 < end && t[i + 1] < 0x80 {
        let c = t[i + 1].to_ascii_lowercase();
        match c {
            b's' | b't' | b'm' | b'd' => return Some(i + 2),
            b'r' | b'v' | b'l' => {
                if i + 2 < end && t[i + 2] < 0x80 {
                    let c2 = t[i + 2].to_ascii_lowercase();
                    if (c == b'r' && c2 == b'e') || (c == b'v' && c2 == b'e') || (c == b'l' && c2 == b'l') { return Some(i + 3); }
                }
            }
            _ => {}
        }
    }
    // Rule 2: [^\r\n\p{L}\p{N}]? \p{L}+   (NEON finds the letter run-end)
    if !nl(b) && !letter(b) && !digit(b) {
        let k = first_not(t, i + 1, end, cl_l);
        if k > i + 1 { if k < end && t[k] >= 0x80 { return None; } return Some(k); }
    }
    if letter(b) {
        let k = first_not(t, i, end, cl_l);
        if k < end && t[k] >= 0x80 { return None; }
        return Some(k);
    }
    // Rule 3: \p{N}{1,3}
    if digit(b) {
        let (mut k, mut cnt) = (i + 1, 1);
        while k < end && cnt < 3 { let c = t[k]; if c >= 0x80 { return None; } if digit(c) { k += 1; cnt += 1; } else { break; } }
        return Some(k);
    }
    // Rule 4:  ?[^\s\p{L}\p{N}]+[\r\n]*   (NEON finds the punct run-end)
    {
        let sp = if b == b' ' { i + 1 } else { i };
        let k = first_not(t, sp, end, cl_p);
        if k > sp {
            if k < end && t[k] >= 0x80 { return None; }
            let mut e = k;
            while e < end { let c = t[e]; if c >= 0x80 { break; } if nl(c) { e += 1; } else { break; } }
            return Some(e);
        }
    }
    // Rules 5/6: pure-space run (no newline). NEON finds the run-end.
    if b == b' ' {
        let m = first_not(t, i, end, cl_s);
        if m == end { return Some(m); }                  // rule6: whole run at end of text
        let y = t[m];
        if y < 0x80 && (letter(y) || (!wsa(y) && !digit(y))) {
            return Some(m - 1);                          // rule6: emit run-1; last space joins next (rule2/4)
        }
        // followed by digit / newline / non-ASCII -> onig fallback (gap semantics)
    }
    None
}

// ---- split-only (pre-tokenization) counters, for the splitter shootout ----
fn split_onig(t: &[u8]) -> usize {
    let s = std::str::from_utf8(t).unwrap();
    let (mut off, mut n) = (0usize, 0usize);
    while off < s.len() {
        match RE.with(|re| re.find(&s[off..])) {
            Some((a, b)) => { if a > 0 { n += 1; } n += 1; off += if b > 0 { b } else { a + 1 }; }
            None => { n += 1; break; }
        }
    }
    n
}
fn split_pcre2(t: &[u8], re: &pcre2::bytes::Regex) -> usize {
    // fastokens' loop: find_at(pos) reusing the regex, no per-match iterator/Result alloc churn
    let (mut n, mut pos) = (0usize, 0usize);
    while pos < t.len() {
        match re.find_at(t, pos) {
            Ok(Some(m)) => { if m.start() == m.end() { pos = m.end() + 1; continue; } n += 1; pos = m.end(); }
            Ok(None) => break,
            Err(_) => break,
        }
    }
    n
}
fn split_dfa(t: &[u8], a: &Cls, b: &Cls, c: &Cls) -> usize {
    let (mut n, mut i) = (0usize, 0usize);
    while i < t.len() {
        if let Some(j) = fast_token(t, i, t.len(), a, b, c) { n += 1; i = j; }
        else {
            let s = unsafe { std::str::from_utf8_unchecked(&t[i..]) };
            match RE.with(|re| re.find(s)) {
                Some((x, y)) => { if x > 0 { n += 1; } n += 1; i += if y > 0 { y } else { x + 1 }; }
                None => { n += 1; break; }
            }
        }
    }
    n
}

struct Encoder { bpe: Bpe, sp: Specials, cl_l: Cls, cl_p: Cls, cl_s: Cls }
impl Encoder {
    fn encode<const L: u8>(&self, input: &[u8], out: &mut Vec<u32>, map: &mut Vec<u8>, syms: &mut Vec<u32>) {
        out.clear();
        let mut pos = 0usize;
        while pos < input.len() {
            match self.sp.next(input, pos) {
                Some((s, len, id)) => {
                    if L >= 2 && s > pos { self.region::<L>(input, pos, s, out, map, syms); }
                    if L >= 5 { out.push(id); }
                    pos = s + len;
                }
                None => { if L >= 2 { self.region::<L>(input, pos, input.len(), out, map, syms); } break; }
            }
        }
    }
    #[inline]
    fn region<const L: u8>(&self, input: &[u8], start: usize, end: usize, out: &mut Vec<u32>, map: &mut Vec<u8>, syms: &mut Vec<u32>) {
        if start >= end { return; }
        let s = std::str::from_utf8(&input[start..end]).unwrap();
        let mut off = 0usize;
        while off < s.len() {
            match RE.with(|re| re.find(&s[off..])) {
                Some((a, b)) => {
                    if a > 0 { self.bpe.piece::<L>(&input[start + off..start + off + a], out, map, syms); }
                    self.bpe.piece::<L>(&input[start + off + a..start + off + b], out, map, syms);
                    off += if b > 0 { b } else { a + 1 };
                }
                None => { self.bpe.piece::<L>(&input[start + off..end], out, map, syms); break; }
            }
        }
    }

    // DFA pre-tokenizer with onig fallback (replaces the regex split in `region`).
    fn encode_dfa<const L: u8>(&self, input: &[u8], out: &mut Vec<u32>, map: &mut Vec<u8>, syms: &mut Vec<u32>) {
        out.clear();
        let mut pos = 0usize;
        while pos < input.len() {
            match self.sp.next(input, pos) {
                Some((s, len, id)) => {
                    if L >= 2 && s > pos { self.region_dfa::<L>(input, pos, s, out, map, syms); }
                    if L >= 5 { out.push(id); }
                    pos = s + len;
                }
                None => { if L >= 2 { self.region_dfa::<L>(input, pos, input.len(), out, map, syms); } break; }
            }
        }
    }
    #[inline]
    fn region_dfa<const L: u8>(&self, input: &[u8], start: usize, end: usize, out: &mut Vec<u32>, map: &mut Vec<u8>, syms: &mut Vec<u32>) {
        let mut i = start;
        while i < end {
            if let Some(j) = fast_token(input, i, end, &self.cl_l, &self.cl_p, &self.cl_s) {
                self.bpe.piece::<L>(&input[i..j], out, map, syms);
                i = j;
            } else {
                // onig fallback for exactly one token at i (non-ASCII / whitespace-run)
                let s = unsafe { std::str::from_utf8_unchecked(&input[i..end]) };
                match RE.with(|re| re.find(s)) {
                    Some((a, b)) => {
                        if a > 0 { self.bpe.piece::<L>(&input[i..i + a], out, map, syms); }
                        self.bpe.piece::<L>(&input[i + a..i + b], out, map, syms);
                        i += if b > 0 { b } else { a + 1 };
                    }
                    None => { self.bpe.piece::<L>(&input[i..end], out, map, syms); break; }
                }
            }
        }
    }
    // count fast-handled vs onig-fallback pre-tokens (over the non-special spans)
    fn fallback_stats(&self, input: &[u8]) -> (u64, u64) {
        let (mut fast, mut fb) = (0u64, 0u64);
        let mut pos = 0usize;
        let mut scan = |start: usize, end: usize, fast: &mut u64, fb: &mut u64| {
            let mut i = start;
            while i < end {
                if let Some(j) = fast_token(input, i, end, &self.cl_l, &self.cl_p, &self.cl_s) { *fast += 1; i = j; }
                else {
                    let s = unsafe { std::str::from_utf8_unchecked(&input[i..end]) };
                    match RE.with(|re| re.find(s)) { Some((_, b)) => { *fb += 1; i += if b > 0 { b } else { 1 }; } None => break }
                }
            }
        };
        while pos < input.len() {
            match self.sp.next(input, pos) {
                Some((s, _len, _id)) => { if s > pos { scan(pos, s, &mut fast, &mut fb); } pos = s + _len; }
                None => { scan(pos, input.len(), &mut fast, &mut fb); break; }
            }
        }
        (fast, fb)
    }

    // optimized encode: NEON-DFA split + raw-byte vocab (no byte-map) + heap merge.
    // STAGE PROFILING (DFA path): split-only, then split+vocab-lookup (no merge).
    fn stage_split(&self, input: &[u8]) -> usize {
        let (mut i, end, mut n) = (0usize, input.len(), 0usize);
        while i < end {
            if let Some(j) = fast_token(input, i, end, &self.cl_l, &self.cl_p, &self.cl_s) { n += 1; i = j; }
            else { let s = unsafe { std::str::from_utf8_unchecked(&input[i..end]) };
                match RE.with(|re| re.find(s)) { Some((a, b)) => { if a > 0 { n += 1; } n += 1; i += if b > 0 { b } else { a + 1 }; } None => { n += 1; break; } } }
        }
        n
    }
    fn stage_lookup(&self, input: &[u8], out: &mut Vec<u32>) {
        out.clear();
        let (mut i, end) = (0usize, input.len());
        while i < end {
            let j = match fast_token(input, i, end, &self.cl_l, &self.cl_p, &self.cl_s) { Some(j) => j, None => { i += 1; continue; } };
            let p = &input[i..j];
            if let Some(&id) = self.bpe.vocab_raw.get(p) { out.push(id); } else { for &b in p { out.push(self.bpe.byte_id[b as usize]); } }
            i = j;
        }
    }
    fn stage_lookup_mphf(&self, input: &[u8], out: &mut Vec<u32>) {
        out.clear();
        let (mut i, end) = (0usize, input.len());
        let vs = self.bpe.vs.as_ref().unwrap();
        while i < end {
            let j = match fast_token(input, i, end, &self.cl_l, &self.cl_p, &self.cl_s) { Some(j) => j, None => { i += 1; continue; } };
            let p = &input[i..j];
            if let Some(id) = vs.get(p) { out.push(id); } else { for &b in p { out.push(self.bpe.byte_id[b as usize]); } }
            i = j;
        }
    }
    fn encode_dfa5(&self, input: &[u8], out: &mut Vec<u32>, syms: &mut Vec<u32>, ms: &mut MergeScratch) {
        out.clear();
        if out.capacity() < input.len() / 3 { out.reserve(input.len() / 3); } // pre-size output (~3 B/token)
        let mut pos = 0usize;
        while pos < input.len() {
            match self.sp.next(input, pos) {
                Some((s, len, id)) => {
                    if s > pos { self.region_dfa5(input, pos, s, out, syms, ms); }
                    out.push(id);
                    pos = s + len;
                }
                None => { self.region_dfa5(input, pos, input.len(), out, syms, ms); break; }
            }
        }
    }
    // FINAL path: NEON-DFA split + thread-local FlatCache (pretoken->ids) + hybrid merge on miss.
    fn encode_cached(&self, input: &[u8], out: &mut Vec<u32>, syms: &mut Vec<u32>, ms: &mut MergeScratch, cache: &mut FlatCache) {
        out.clear();
        if out.capacity() < input.len() / 3 { out.reserve(input.len() / 3); }
        let mut pos = 0usize;
        while pos < input.len() {
            match self.sp.next(input, pos) {
                Some((s, len, id)) => { if s > pos { self.region_cached(input, pos, s, out, syms, ms, cache); } out.push(id); pos = s + len; }
                None => { self.region_cached(input, pos, input.len(), out, syms, ms, cache); break; }
            }
        }
    }
    #[inline]
    fn region_cached(&self, input: &[u8], start: usize, end: usize, out: &mut Vec<u32>, syms: &mut Vec<u32>, ms: &mut MergeScratch, cache: &mut FlatCache) {
        let mut i = start;
        while i < end {
            if let Some(j) = fast_token(input, i, end, &self.cl_l, &self.cl_p, &self.cl_s) {
                let p = &input[i..j];
                let h = cache.hasher.hash_one(p);
                if let Some((off, len)) = cache.get(p, h) {
                    out.extend_from_slice(&cache.ids[off as usize..off as usize + len as usize]);
                } else {
                    let st = out.len();
                    self.bpe.piece5(p, out, syms, ms);
                    cache.insert(p, h, &out[st..]);
                }
                i = j;
            } else {
                let s = unsafe { std::str::from_utf8_unchecked(&input[i..end]) };
                match RE.with(|re| re.find(s)) {
                    Some((a, b)) => { if a > 0 { self.bpe.piece5(&input[i..i + a], out, syms, ms); }
                        self.bpe.piece5(&input[i + a..i + b], out, syms, ms); i += if b > 0 { b } else { a + 1 }; }
                    None => { self.bpe.piece5(&input[i..end], out, syms, ms); break; }
                }
            }
        }
    }
    // full encode using the MPHF VocabStore for bytes->id (vs AHashMap)
    fn encode_dfa5_mphf(&self, input: &[u8], out: &mut Vec<u32>, syms: &mut Vec<u32>, ms: &mut MergeScratch) {
        out.clear();
        if out.capacity() < input.len() / 3 { out.reserve(input.len() / 3); }
        let (mut i, end) = (0usize, input.len());
        while i < end {
            if let Some(j) = fast_token(input, i, end, &self.cl_l, &self.cl_p, &self.cl_s) {
                self.bpe.piece5_mphf(&input[i..j], out, syms, ms); i = j;
            } else {
                let s = unsafe { std::str::from_utf8_unchecked(&input[i..end]) };
                match RE.with(|re| re.find(s)) {
                    Some((a, b)) => { if a > 0 { self.bpe.piece5_mphf(&input[i..i + a], out, syms, ms); }
                        self.bpe.piece5_mphf(&input[i + a..i + b], out, syms, ms); i += if b > 0 { b } else { a + 1 }; }
                    None => { self.bpe.piece5_mphf(&input[i..end], out, syms, ms); break; }
                }
            }
        }
    }
    #[inline]
    fn region_dfa5(&self, input: &[u8], start: usize, end: usize, out: &mut Vec<u32>, syms: &mut Vec<u32>, ms: &mut MergeScratch) {
        let mut i = start;
        while i < end {
            if let Some(j) = fast_token(input, i, end, &self.cl_l, &self.cl_p, &self.cl_s) {
                self.bpe.piece5(&input[i..j], out, syms, ms);
                i = j;
            } else {
                let s = unsafe { std::str::from_utf8_unchecked(&input[i..end]) };
                match RE.with(|re| re.find(s)) {
                    Some((a, b)) => {
                        if a > 0 { self.bpe.piece5(&input[i..i + a], out, syms, ms); }
                        self.bpe.piece5(&input[i + a..i + b], out, syms, ms);
                        i += if b > 0 { b } else { a + 1 };
                    }
                    None => { self.bpe.piece5(&input[i..end], out, syms, ms); break; }
                }
            }
        }
    }

    // recording variants: append per-unit (byte_end, id_count) so a later call can reuse a prefix.
    #[inline]
    fn region_record(&self, input: &[u8], start: usize, end: usize, out: &mut Vec<u32>, syms: &mut Vec<u32>, ms: &mut MergeScratch, ends: &mut Vec<(u32, u32)>) {
        let mut i = start;
        while i < end {
            if let Some(j) = fast_token(input, i, end, &self.cl_l, &self.cl_p, &self.cl_s) {
                self.bpe.piece5(&input[i..j], out, syms, ms);
                ends.push((j as u32, out.len() as u32));
                i = j;
            } else {
                let s = unsafe { std::str::from_utf8_unchecked(&input[i..end]) };
                match RE.with(|re| re.find(s)) {
                    Some((a, b)) => {
                        if a > 0 { self.bpe.piece5(&input[i..i + a], out, syms, ms); ends.push(((i + a) as u32, out.len() as u32)); }
                        self.bpe.piece5(&input[i + a..i + b], out, syms, ms); ends.push(((i + b) as u32, out.len() as u32));
                        i += if b > 0 { b } else { a + 1 };
                    }
                    None => { self.bpe.piece5(&input[i..end], out, syms, ms); ends.push((end as u32, out.len() as u32)); break; }
                }
            }
        }
    }
    fn encode_record(&self, input: &[u8], start: usize, out: &mut Vec<u32>, syms: &mut Vec<u32>, ms: &mut MergeScratch, ends: &mut Vec<(u32, u32)>) {
        let mut pos = start;
        while pos < input.len() {
            match self.sp.next(input, pos) {
                Some((s, len, id)) => {
                    if s > pos { self.region_record(input, pos, s, out, syms, ms, ends); }
                    out.push(id);
                    ends.push(((s + len) as u32, out.len() as u32));
                    pos = s + len;
                }
                None => { self.region_record(input, pos, input.len(), out, syms, ms, ends); break; }
            }
        }
    }
    // Multi-turn re-encode: reuse the unchanged prefix's tokens, re-encode only the new suffix.
    fn encode_incremental(&self, input: &[u8], c: &mut PrefixCache, out: &mut Vec<u32>, syms: &mut Vec<u32>, ms: &mut MergeScratch) {
        out.clear();
        let common = common_prefix_len(&c.prev, input);
        // reuse units fully before `common` (their 1-char lookahead stays inside the shared region)
        let reuse = c.ends.partition_point(|&(be, _)| (be as usize) < common);
        let (restart, id_reuse) = if reuse > 0 { let (be, ie) = c.ends[reuse - 1]; (be as usize, ie as usize) } else { (0, 0) };
        out.extend_from_slice(&c.ids[..id_reuse]);
        c.ends.truncate(reuse);
        self.encode_record(input, restart, out, syms, ms, &mut c.ends);
        c.prev.clear(); c.prev.extend_from_slice(input);
        c.ids.clear(); c.ids.extend_from_slice(out);
    }

    // batched-vs-interleaved test (no specials; big.txt has ~none). Interleaved = split one + BPE one.
    fn encode_interleaved_ns(&self, input: &[u8], out: &mut Vec<u32>, map: &mut Vec<u8>, syms: &mut Vec<u32>) {
        out.clear();
        self.region_dfa::<5>(input, 0, input.len(), out, map, syms);
    }
    // Batched = phase 1 split the whole buffer into ranges, phase 2 BPE all ranges (decoupled loops).
    fn encode_batched_ns(&self, input: &[u8], ranges: &mut Vec<(u32, u32)>, out: &mut Vec<u32>, map: &mut Vec<u8>, syms: &mut Vec<u32>) {
        out.clear();
        ranges.clear();
        let mut i = 0usize;
        while i < input.len() {
            if let Some(j) = fast_token(input, i, input.len(), &self.cl_l, &self.cl_p, &self.cl_s) {
                ranges.push((i as u32, j as u32)); i = j;
            } else {
                let s = unsafe { std::str::from_utf8_unchecked(&input[i..]) };
                match RE.with(|re| re.find(s)) {
                    Some((a, b)) => {
                        if a > 0 { ranges.push((i as u32, (i + a) as u32)); }
                        ranges.push(((i + a) as u32, (i + b) as u32));
                        i += if b > 0 { b } else { a + 1 };
                    }
                    None => { ranges.push((i as u32, input.len() as u32)); break; }
                }
            }
        }
        for &(a, b) in ranges.iter() { self.bpe.piece::<5>(&input[a as usize..b as usize], out, map, syms); }
    }

    // IREE-style ring: stream input through a fixed buffer; only commit complete tokens, carry the tail.
    fn encode_ring(&self, input: &[u8], ring: &mut [u8], out: &mut Vec<u32>, map: &mut Vec<u8>, syms: &mut Vec<u32>) {
        out.clear();
        let r = ring.len();
        let (mut filled, mut in_pos) = (0usize, 0usize);
        loop {
            let take = (r - filled).min(input.len() - in_pos);
            ring[filled..filled + take].copy_from_slice(&input[in_pos..in_pos + take]);
            in_pos += take; filled += take;
            let last = in_pos == input.len();
            let consumed = self.window(&ring[..filled], out, map, syms, last);
            if last && consumed >= filled { break; }
            ring.copy_within(consumed..filled, 0);
            filled -= consumed;
            if last && filled == 0 { break; }
            debug_assert!(consumed > 0 || last, "ring too small for a pretoken");
            if consumed == 0 && !last { /* pretoken longer than ring: force-emit to make progress */
                self.region::<5>(&ring[..filled], 0, filled, out, map, syms); filled = 0;
            }
        }
    }
    // process ring window; commit only tokens fully inside unless `last`; return bytes consumed.
    fn window(&self, w: &[u8], out: &mut Vec<u32>, map: &mut Vec<u8>, syms: &mut Vec<u32>, last: bool) -> usize {
        let mut pos = 0usize;
        loop {
            match self.sp.next(w, pos) {
                Some((s, len, id)) => {
                    if s > pos { self.region::<5>(w, pos, s, out, map, syms); } // gap end s is a real boundary
                    if s + len <= w.len() { out.push(id); pos = s + len; }
                    else if last { out.push(id); return w.len(); }
                    else { return s; } // special straddles window edge -> carry from s
                }
                None => {
                    // trailing region: defer the final regex token unless this is the last window
                    return pos + self.region_commit(&w[pos..], out, map, syms, last);
                }
            }
        }
    }
    // BPE a region, deferring the last regex match if it touches the end (unless can_finish). Returns consumed.
    fn region_commit(&self, w: &[u8], out: &mut Vec<u32>, map: &mut Vec<u8>, syms: &mut Vec<u32>, can_finish: bool) -> usize {
        if w.is_empty() { return 0; }
        // ring window may end mid-UTF8-char: only scan the valid prefix; the partial tail is carried.
        let valid = match std::str::from_utf8(w) { Ok(s) => s.len(), Err(e) => e.valid_up_to() };
        if valid == 0 { return 0; }
        let s = unsafe { std::str::from_utf8_unchecked(&w[..valid]) };
        let mut off = 0usize;
        let mut pending: Option<(usize, usize)> = None;
        loop {
            match RE.with(|re| re.find(&s[off..])) {
                Some((a, b)) => {
                    let (aa, ab) = (off + a, off + b);
                    if let Some((pa, pb)) = pending { self.bpe.piece::<5>(&w[pa..pb], out, map, syms); }
                    pending = Some((aa, ab));
                    off += if b > 0 { b } else { a + 1 };
                    if off >= s.len() { break; }
                }
                None => break,
            }
        }
        match pending {
            // defer the last token unless this is the final window (it may extend past the edge)
            Some((pa, pb)) => { if can_finish { self.bpe.piece::<5>(&w[pa..pb], out, map, syms); pb } else { pa } }
            None => if can_finish { w.len() } else { valid },
        }
    }
}

fn build_encoder(tok_path: &str) -> Encoder {
    let bpe = Bpe::load(tok_path);
    let j: Value = serde_json::from_slice(&std::fs::read(tok_path).unwrap()).unwrap();
    let specials: Vec<(Vec<u8>, u32)> = j["added_tokens"].as_array().map(|a| a.iter()
        .map(|a| (a["content"].as_str().unwrap().as_bytes().to_vec(), a["id"].as_u64().unwrap() as u32)).collect()).unwrap_or_default();
    let cl_l = build_cls(|c| letter(c));
    let cl_p = build_cls(|c| c < 0x80 && !wsa(c) && !letter(c) && !digit(c));
    let cl_s = build_cls(|c| c == b' ');
    Encoder { bpe, sp: Specials::new(&specials), cl_l, cl_p, cl_s }
}

// Data paths resolved relative to the crate (location-independent; works from any cwd).
// big.txt / llama-3 live in the repo's tokenizers/data; the 22-model sweep tokenizers go in
// poc/data/toks/ (fetch with `python poc/scripts/download_tokenizers.py`).
fn data_file(name: &str) -> String { format!("{}/../../tokenizers/data/{}", env!("CARGO_MANIFEST_DIR"), name) }
fn workloads_path() -> String { format!("{}/workloads.json", env!("CARGO_MANIFEST_DIR")) }
fn tok_path(model: &str) -> String { format!("{}/../data/toks/{}.json", env!("CARGO_MANIFEST_DIR"), model) }

// ====== FINAL POC SWEEP: model x task x {single, multi-thread}, cached ======
fn sweep() {
    use rayon::prelude::*;
    // Group-A models the POC is byte-exact on (llama3/GPT-4 split regex + GPT-2 byte-level)
    let models = ["llama3", "llama3.1", "llama3.3", "smollm3", "deepseek-r1-llama", "deepseek-r1-qwen",
        "qwen2.5-7b", "qwen3", "qwq", "qwen2.5-vl", "olmo2", "phi4"];
    let w: Value = serde_json::from_slice(&std::fs::read(workloads_path()).unwrap()).unwrap();
    let big = std::fs::read_to_string(data_file("big.txt")).unwrap();
    let cols = ["thinking", "paste", "conv+ctx", "conv", "dense", "plain-en"];

    // single-thread cached MB/s for one text
    let st = |enc: &Encoder, t: &[u8]| -> f64 {
        let (mut o, mut sy, mut ms, mut c) = (Vec::new(), Vec::new(), MergeScratch::new(), FlatCache::new());
        let iters = (8_000_000 / t.len().max(1)).clamp(50, 8000) as u64;
        for _ in 0..5 { enc.encode_cached(black_box(t), &mut o, &mut sy, &mut ms, &mut c); }
        let s = Instant::now();
        for _ in 0..iters { enc.encode_cached(black_box(t), &mut o, &mut sy, &mut ms, &mut c); black_box(&o); }
        1000.0 / (s.elapsed().as_nanos() as f64 / (iters as usize * t.len()) as f64)
    };

    println!("=== FINAL POC (cached) single-thread MB/s per model x task ===");
    print!("{:<18}", "model");
    for c in &cols { print!("{:>9}", c); }
    println!("{:>10}{:>7}", "8-thrd", "scale");
    for m in &models {
        let enc = build_encoder(&tok_path(m));
        let entry = w.as_array().unwrap().iter().find(|x| x["name"] == *m);
        print!("{:<18}", m);
        for col in &cols {
            let txt: Option<Vec<u8>> = if *col == "plain-en" { Some(big[..40_000].as_bytes().to_vec()) }
                else { entry.and_then(|e| e["items"].as_array().unwrap().iter().find(|it| it["type"] == *col)
                    .map(|it| it["text"].as_str().unwrap().as_bytes().to_vec())) };
            match txt { Some(t) => print!("{:>9.0}", st(&enc, &t)), None => print!("{:>9}", "-") }
        }
        // multi-thread on a 24 MB English corpus, cached, per-thread cache (zero contention)
        let mut corpus = String::new(); while corpus.len() < 24_000_000 { corpus.push_str(&big); }
        let cb = corpus.as_bytes();
        let mut docs: Vec<&str> = Vec::new(); let mut last = 0;
        for (idx, &c) in cb.iter().enumerate() { if c == b'\n' && idx - last >= 8192 { docs.push(&corpus[last..=idx]); last = idx + 1; } }
        if last < corpus.len() { docs.push(&corpus[last..]); }
        let total: usize = docs.iter().map(|d| d.len()).sum();
        let run = |nt: usize| -> f64 {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(nt).build().unwrap();
            let go = || pool.install(|| docs.par_iter().map_init(
                || (Vec::<u32>::new(), Vec::<u32>::new(), MergeScratch::new(), FlatCache::new()),
                |(o, sy, ms, c), d| { enc.encode_cached(d.as_bytes(), o, sy, ms, c); o.len() as u64 }).sum::<u64>());
            let _ = go(); let t = Instant::now(); let _ = go();
            total as f64 / 1e6 / t.elapsed().as_secs_f64()
        };
        let (s1, s8) = (run(1), run(8));
        println!("{:>8.0}MB{:>6.1}x", s8, s8 / s1);
    }
}

// cached vs uncached across thread counts on a 24 MB English corpus (llama3)
fn cmp_threads() {
    use rayon::prelude::*;
    let enc = build_encoder(&data_file("llama-3-tokenizer.json"));
    let big = std::fs::read_to_string(data_file("big.txt")).unwrap();
    let mut corpus = String::new(); while corpus.len() < 24_000_000 { corpus.push_str(&big); }
    let cb = corpus.as_bytes();
    let mut docs: Vec<&str> = Vec::new(); let mut last = 0;
    for (idx, &c) in cb.iter().enumerate() { if c == b'\n' && idx - last >= 8192 { docs.push(&corpus[last..=idx]); last = idx + 1; } }
    if last < corpus.len() { docs.push(&corpus[last..]); }
    let total: usize = docs.iter().map(|d| d.len()).sum();
    let run = |nt: usize, cached: bool| -> f64 {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(nt).build().unwrap();
        let go = || pool.install(|| docs.par_iter().map_init(
            || (Vec::<u32>::new(), Vec::<u32>::new(), MergeScratch::new(), FlatCache::new()),
            |(o, sy, ms, c), d| { if cached { enc.encode_cached(d.as_bytes(), o, sy, ms, c) } else { enc.encode_dfa5(d.as_bytes(), o, sy, ms) } o.len() as u64 }).sum::<u64>());
        let _ = go(); let t = Instant::now(); let _ = go();
        total as f64 / 1e6 / t.elapsed().as_secs_f64()
    };
    println!("=== cached vs uncached x threads (llama3, {:.0} MB repeated-English corpus) ===", total as f64 / 1e6);
    println!("{:<8}{:>12}{:>12}{:>8}", "threads", "uncached", "cached", "cache");
    for nt in [1usize, 2, 4, 8] {
        let (u, c) = (run(nt, false), run(nt, true));
        println!("{:<8}{:>9.0} MB{:>9.0} MB{:>7.2}x", nt, u, c, c / u);
    }
}

fn main() {
    if std::env::var("CMP").is_ok() { cmp_threads(); return; }
    if std::env::var("SWEEP").is_ok() { sweep(); return; }
    let enc = build_encoder(&data_file("llama-3-tokenizer.json"));
    let p = data_file("llama-3-tokenizer.json");
    let j: Value = serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap();
    let _ = &j;

    let w: Value = serde_json::from_slice(&std::fs::read(workloads_path()).unwrap()).unwrap();
    let llama = w.as_array().unwrap().iter().find(|m| m["name"] == "llama3.1").unwrap();
    let big = std::fs::read_to_string(data_file("big.txt")).unwrap();

    // ===== CLEAN DFA-path stage profile + AHashMap vs MPHF bytes->id (big.txt) =====
    {
        let tb = big.as_bytes();
        let mut ms = MergeScratch::new();
        let (mut o, mut sy) = (Vec::new(), Vec::new());
        let mut oa = Vec::new(); let mut ob = Vec::new();
        macro_rules! timed { ($b:block) => {{ for _ in 0..3 { $b } let s = Instant::now(); for _ in 0..20 { $b } s.elapsed().as_nanos() as f64 / (20 * tb.len()) as f64 }}; }
        let split = timed!({ black_box(enc.stage_split(black_box(tb))); });
        let look_a = timed!({ enc.stage_lookup(black_box(tb), &mut oa); black_box(&oa); });
        let look_m = timed!({ enc.stage_lookup_mphf(black_box(tb), &mut ob); black_box(&ob); });
        let full_a = timed!({ enc.encode_dfa5(black_box(tb), &mut o, &mut sy, &mut ms); black_box(&o); });
        let full_m = timed!({ enc.encode_dfa5_mphf(black_box(tb), &mut o, &mut sy, &mut ms); black_box(&o); });
        let mut cache = FlatCache::new();
        let full_c = timed!({ enc.encode_cached(black_box(tb), &mut o, &mut sy, &mut ms, &mut cache); black_box(&o); });
        enc.encode_dfa5(tb, &mut o, &mut sy, &mut ms); let na = o.len();
        enc.encode_dfa5_mphf(tb, &mut o, &mut sy, &mut ms); let nm = o.len();
        enc.encode_cached(tb, &mut o, &mut sy, &mut ms, &mut cache); let nc = o.len();
        println!("=== DFA-path stage profile on big.txt ({:.1} MB), ns/byte ===", tb.len() as f64 / 1e6);
        println!("  split-only            {:>6.2}  ({:>5.0} MB/s)", split, 1000.0 / split);
        println!("  +vocab lookup (ahash) {:>6.2}  -> bytes->id costs {:.2}", look_a, look_a - split);
        println!("  +vocab lookup (MPHF)  {:>6.2}  -> bytes->id costs {:.2}", look_m, look_m - split);
        println!("  FULL  (ahash)         {:>6.2}  ({:>5.0} MB/s)  -> merge costs {:.2}", full_a, 1000.0 / full_a, full_a - look_a);
        println!("  FULL  (MPHF)          {:>6.2}  ({:>5.0} MB/s)  parity {}", full_m, 1000.0 / full_m, if na == nm { "OK" } else { "FAIL" });
        println!("  FULL  (CACHED)        {:>6.2}  ({:>5.0} MB/s)  {:.2}x  parity {}", full_c, 1000.0 / full_c, full_a / full_c, if na == nc { "OK" } else { "FAIL" });
        // PROVE allocation-free: warm the buffers, then count alloc calls across 20 cached encodes.
        for _ in 0..5 { enc.encode_cached(tb, &mut o, &mut sy, &mut ms, &mut cache); }
        let a0 = ALLOCS.load(Relaxed);
        for _ in 0..20 { enc.encode_cached(black_box(tb), &mut o, &mut sy, &mut ms, &mut cache); black_box(&o); }
        println!("  allocation calls during 20 warmed CACHED encodes: {}  ({})",
            ALLOCS.load(Relaxed) - a0, if ALLOCS.load(Relaxed) - a0 == 0 { "ALLOCATION-FREE" } else { "still allocating" });
        if std::env::var("PROF_ONLY").is_ok() { return; }
    }

    let (mut o, mut m, mut sy) = (Vec::new(), Vec::new(), Vec::new());
    let mut ring = vec![0u8; 8192]; // IREE-style fixed transform buffer (power of 2)

    let lvl = |t: &[u8], f: &dyn Fn(&[u8], &mut Vec<u32>, &mut Vec<u8>, &mut Vec<u32>)| -> f64 {
        let mut o = Vec::new(); let mut m = Vec::new(); let mut sy = Vec::new();
        let iters = (2_000_000 / t.len().max(1)).clamp(20, 5000) as u64;
        for _ in 0..3 { f(black_box(t), &mut o, &mut m, &mut sy); }
        let s = Instant::now();
        for _ in 0..iters { f(black_box(t), &mut o, &mut m, &mut sy); black_box(&o); }
        s.elapsed().as_nanos() as f64 / (iters as usize * t.len()) as f64
    };

    let prompts: Vec<(&str, &[u8])> = llama["items"].as_array().unwrap().iter()
        .map(|it| (it["type"].as_str().unwrap(), it["text"].as_str().unwrap().as_bytes()))
        .chain(std::iter::once(("plain-en", big[..40_000.min(big.len())].as_bytes())))
        .collect();

    // correctness: full == ring token counts
    println!("parity (full vs ring token counts):");
    for (name, t) in &prompts {
        enc.encode::<5>(t, &mut o, &mut m, &mut sy); let nf = o.len();
        enc.encode_ring(t, &mut ring, &mut o, &mut m, &mut sy); let nr = o.len();
        println!("  {:<10} full={:<6} ring={:<6} {}", name, nf, nr, if nf == nr { "OK" } else { "MISMATCH" });
    }

    println!("\nper-stage breakdown (ns/byte, cumulative levels then deltas):");
    println!("{:<10} {:>7} {:>7} {:>7} {:>7} {:>7} | {:>6} {:>6} {:>6} {:>6} {:>6}",
        "prompt", "L1mat", "L2spl", "L3map", "L4voc", "L5full", "match", "split", "map", "vocab", "merge");
    for (name, t) in &prompts {
        let l1 = lvl(t, &|t, o, m, s| enc.encode::<1>(t, o, m, s));
        let l2 = lvl(t, &|t, o, m, s| enc.encode::<2>(t, o, m, s));
        let l3 = lvl(t, &|t, o, m, s| enc.encode::<3>(t, o, m, s));
        let l4 = lvl(t, &|t, o, m, s| enc.encode::<4>(t, o, m, s));
        let l5 = lvl(t, &|t, o, m, s| enc.encode::<5>(t, o, m, s));
        println!("{:<10} {:>7.2} {:>7.2} {:>7.2} {:>7.2} {:>7.2} | {:>6.2} {:>6.2} {:>6.2} {:>6.2} {:>6.2}",
            name, l1, l2, l3, l4, l5,
            l1, (l2 - l1).max(0.0), (l3 - l2).max(0.0), (l4 - l3).max(0.0), (l5 - l4).max(0.0));
    }

    println!("\nDFA pre-tokenizer vs onig (full encode, ns/byte) + parity + fallback rate:");
    println!("{:<10} {:>8} {:>8} {:>7} | {:>8} {:>8} {:>7}", "prompt", "onig", "DFA", "speedup", "tok", "tokDFA", "fallbk%");
    for (name, t) in &prompts {
        enc.encode::<5>(t, &mut o, &mut m, &mut sy); let n_onig = o.len();
        enc.encode_dfa::<5>(t, &mut o, &mut m, &mut sy); let n_dfa = o.len();
        let onig = lvl(t, &|t, o, m, s| enc.encode::<5>(t, o, m, s));
        let dfa = lvl(t, &|t, o, m, s| enc.encode_dfa::<5>(t, o, m, s));
        let (fast, fb) = enc.fallback_stats(t);
        println!("{:<10} {:>6.2}ns {:>6.2}ns {:>6.1}x | {:>8} {:>8} {:>6.1}% {}",
            name, onig, dfa, onig / dfa, n_onig, n_dfa, 100.0 * fb as f64 / (fast + fb).max(1) as f64,
            if n_onig == n_dfa { "" } else { "PARITY FAIL" });
    }

    println!("\nflat vs ring (ns/byte):");
    for (name, t) in &prompts {
        let flat = lvl(t, &|t, o, m, s| enc.encode::<5>(t, o, m, s));
        let r = {
            let mut o = Vec::new(); let mut m = Vec::new(); let mut sy = Vec::new(); let mut ring = vec![0u8; 8192];
            let iters = (2_000_000 / t.len().max(1)).clamp(20, 5000) as u64;
            for _ in 0..3 { enc.encode_ring(black_box(t), &mut ring, &mut o, &mut m, &mut sy); }
            let s = Instant::now();
            for _ in 0..iters { enc.encode_ring(black_box(t), &mut ring, &mut o, &mut m, &mut sy); black_box(&o); }
            s.elapsed().as_nanos() as f64 / (iters as usize * t.len()) as f64
        };
        println!("  {:<10} flat {:>6.2}  ring(8K) {:>6.2}", name, flat, r);
    }

    // ---- parallel scaling (rayon, DFA path) ----
    use rayon::prelude::*;
    let mut corpus = String::new();
    while corpus.len() < 48_000_000 { corpus.push_str(&big); }
    let cb = corpus.as_bytes();
    let mut docs: Vec<&str> = Vec::new();
    let mut last = 0usize;
    for (idx, &c) in cb.iter().enumerate() {
        if c == b'\n' && idx - last >= 8192 { docs.push(&corpus[last..=idx]); last = idx + 1; }
    }
    if last < corpus.len() { docs.push(&corpus[last..]); }
    let total: usize = docs.iter().map(|d| d.len()).sum();
    let run = |nt: usize| -> (f64, u64) {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(nt).build().unwrap();
        let go = || pool.install(|| docs.par_iter().map_init(
            || (Vec::<u32>::new(), Vec::<u32>::new(), MergeScratch::new()),
            |(o, syms, ms), d| { enc.encode_dfa5(d.as_bytes(), o, syms, ms); o.len() as u64 },
        ).sum::<u64>());
        let _ = go();
        let t = Instant::now();
        let toks = go();
        (total as f64 / 1e6 / t.elapsed().as_secs_f64(), toks)
    };
    println!("\nparallel scaling (rayon, DFA encode, NO cache) on {:.0} MB, {} docs:", total as f64 / 1e6, docs.len());
    let (base, _) = run(1);
    for nt in [1usize, 2, 4, 8] {
        let (mbps, toks) = run(nt);
        println!("  threads={:<2} {:>7.0} MB/s   {:>4.2}x   {:>3.0}% eff   ({} tok)",
            nt, mbps, mbps / base, 100.0 * mbps / base / nt as f64, toks);
    }

    // ============ pre-tokenization shootout (the real bottleneck) ============
    let tb = big.as_bytes();
    let re2 = pcre2::bytes::RegexBuilder::new().jit(true).utf(true).ucp(true).build(PATTERN).unwrap();
    let split_time = |f: &dyn Fn() -> usize| -> (f64, usize) {
        for _ in 0..3 { black_box(f()); }
        let s = Instant::now();
        let mut n = 0;
        for _ in 0..20 { n = f(); }
        (s.elapsed().as_nanos() as f64 / (20 * tb.len()) as f64, n)
    };
    println!("\nSPLIT-ONLY (pre-tokenization) on big.txt ({:.1} MB):", tb.len() as f64 / 1e6);
    let (o_ns, o_n) = split_time(&|| split_onig(tb));
    let (p_ns, p_n) = split_time(&|| split_pcre2(tb, &re2));
    let (d_ns, d_n) = split_time(&|| split_dfa(tb, &enc.cl_l, &enc.cl_p, &enc.cl_s));
    println!("  onig       {:>6.2} ns/byte  ({:>5.0} MB/s)  {} pretok", o_ns, 1000.0 / o_ns, o_n);
    println!("  pcre2-JIT  {:>6.2} ns/byte  ({:>5.0} MB/s)  {} pretok", p_ns, 1000.0 / p_ns, p_n);
    println!("  NEON-DFA   {:>6.2} ns/byte  ({:>5.0} MB/s)  {} pretok", d_ns, 1000.0 / d_ns, d_n);

    // batched vs interleaved (full DFA encode, no specials)
    let (mut o, mut m, mut s, mut rg) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    enc.encode_interleaved_ns(tb, &mut o, &mut m, &mut s); let n_i = o.len();
    enc.encode_batched_ns(tb, &mut rg, &mut o, &mut m, &mut s); let n_b = o.len();
    let full = |batched: bool| -> f64 {
        let (mut o, mut m, mut s, mut rg) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for _ in 0..3 { if batched { enc.encode_batched_ns(tb, &mut rg, &mut o, &mut m, &mut s) } else { enc.encode_interleaved_ns(tb, &mut o, &mut m, &mut s) } }
        let t = Instant::now();
        for _ in 0..20 { if batched { enc.encode_batched_ns(black_box(tb), &mut rg, &mut o, &mut m, &mut s) } else { enc.encode_interleaved_ns(black_box(tb), &mut o, &mut m, &mut s) } black_box(&o); }
        t.elapsed().as_nanos() as f64 / (20 * tb.len()) as f64
    };
    let il = full(false);
    let ba = full(true);
    println!("\nbatched vs interleaved (full DFA encode, big.txt):");
    println!("  interleaved {:>6.2} ns/byte ({:>5.0} MB/s)", il, 1000.0 / il);
    println!("  batched     {:>6.2} ns/byte ({:>5.0} MB/s)  {:.2}x  {}", ba, 1000.0 / ba, il / ba, if n_i == n_b { "" } else { "PARITY FAIL" });

    // ============ optimized BPE path: raw-byte vocab (no byte-map) + heap merge ============
    println!("\nOPTIMIZED full encode (raw-byte vocab + heap merge) vs old DFA, ns/byte:");
    let mut ms = MergeScratch::new();
    for (name, t) in &prompts {
        let tb = *t;
        enc.encode_dfa::<5>(tb, &mut o, &mut m, &mut s); let n_old = o.len();
        enc.encode_dfa5(tb, &mut o, &mut s, &mut ms); let n_new = o.len();
        let iters = (40_000_000 / tb.len().max(1)).clamp(20, 4000) as u64;
        for _ in 0..3 { enc.encode_dfa::<5>(black_box(tb), &mut o, &mut m, &mut s); }
        let t0 = Instant::now(); for _ in 0..iters { enc.encode_dfa::<5>(black_box(tb), &mut o, &mut m, &mut s); black_box(&o); }
        let old = t0.elapsed().as_nanos() as f64 / (iters as usize * tb.len()) as f64;
        for _ in 0..3 { enc.encode_dfa5(black_box(tb), &mut o, &mut s, &mut ms); }
        let t1 = Instant::now(); for _ in 0..iters { enc.encode_dfa5(black_box(tb), &mut o, &mut s, &mut ms); black_box(&o); }
        let new = t1.elapsed().as_nanos() as f64 / (iters as usize * tb.len()) as f64;
        println!("  {:<10} old {:>6.2}ns ({:>5.0} MB/s)  opt {:>6.2}ns ({:>6.0} MB/s)  {:.2}x  {}",
            name, old, 1000.0 / old, new, 1000.0 / new, old / new, if n_old == n_new { "" } else { "PARITY FAIL" });
    }

    // ============ multi-turn chat re-encode: prefix cache vs full re-encode ============
    let conv = big.as_bytes();
    let chunk = 4096usize;
    let turns = (conv.len() / chunk).min(256);
    let mut so = Vec::new();
    // parity: incremental final == full final
    let len_last = turns * chunk;
    enc.encode_dfa5(&conv[..len_last], &mut so, &mut s, &mut ms); let n_full = so.len();
    let mut pc = PrefixCache::new();
    for tt in 1..=turns { enc.encode_incremental(&conv[..tt * chunk], &mut pc, &mut so, &mut s, &mut ms); }
    let n_inc = so.len();
    // time: full re-encode of the whole conversation each turn (O(N^2))
    let t0 = Instant::now();
    for tt in 1..=turns { enc.encode_dfa5(&conv[..tt * chunk], &mut so, &mut s, &mut ms); black_box(&so); }
    let full_ms = t0.elapsed().as_secs_f64() * 1000.0;
    // time: incremental (prefix cache) each turn (O(N))
    let mut pc = PrefixCache::new();
    let t1 = Instant::now();
    for tt in 1..=turns { enc.encode_incremental(&conv[..tt * chunk], &mut pc, &mut so, &mut s, &mut ms); black_box(&so); }
    let inc_ms = t1.elapsed().as_secs_f64() * 1000.0;
    println!("\nmulti-turn chat re-encode ({} turns, +{} B/turn, full conversation encoded each turn):", turns, chunk);
    println!("  full re-encode  : {:>8.1} ms", full_ms);
    println!("  prefix cache    : {:>8.1} ms   {:.1}x faster   parity {}", inc_ms, full_ms / inc_ms, if n_full == n_inc { "OK" } else { "FAIL" });
}
