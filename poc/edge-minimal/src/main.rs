// Cleaned inference-only tk-encode split: NO onig/pcre2 (scalar fallback), NO ahash (FxHash),
// NO rayon, NO offsets/training, no Debug/formatting in the lib path. Deps: memchr + serde_json (load).
// Measures: binary size (build) + loaded-structure memory (counting allocator) + correctness on big.txt.
use serde_json::Value;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::hash::{BuildHasherDefault, Hasher};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::time::Instant;
mod dfa;
use dfa::{build_cls, first_not, Cls};

struct Counting;
static LIVE: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 { let p = System.alloc(l); if !p.is_null() { LIVE.fetch_add(l.size(), Relaxed); } p }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 { let p = System.alloc_zeroed(l); if !p.is_null() { LIVE.fetch_add(l.size(), Relaxed); } p }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) { System.dealloc(p, l); LIVE.fetch_sub(l.size(), Relaxed); }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 { let q = System.realloc(p, l, n); if !q.is_null() { LIVE.fetch_add(n, Relaxed); LIVE.fetch_sub(l.size(), Relaxed); } q }
}
#[global_allocator]
static A: Counting = Counting;
fn live() -> usize { LIVE.load(Relaxed) }

// tiny FxHash (no ahash dep)
#[derive(Default)]
struct Fx(u64);
impl Hasher for Fx {
    fn finish(&self) -> u64 { self.0 }
    fn write(&mut self, b: &[u8]) { let mut h = self.0; for &x in b { h = (h.rotate_left(5) ^ x as u64).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95); } self.0 = h; }
    fn write_u32(&mut self, x: u32) { self.0 = (self.0.rotate_left(5) ^ x as u64).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95); }
}
type Map<K, V> = HashMap<K, V, BuildHasherDefault<Fx>>;
fn newmap<K, V>() -> Map<K, V> { HashMap::default() }

const PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

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
#[inline] fn letter(c: u8) -> bool { c.is_ascii_alphabetic() }
#[inline] fn digit(c: u8) -> bool { c.is_ascii_digit() }
#[inline] fn nl(c: u8) -> bool { c == b'\n' || c == b'\r' }
#[inline] fn wsa(c: u8) -> bool { c == b' ' || (0x09..=0x0d).contains(&c) }

struct Bpe { vocab_raw: Map<Vec<u8>, u32>, ranks: Map<(u32, u32), (u32, u32)>, byte_id: [u32; 256] }
struct MS { next: Vec<i32>, prev: Vec<i32>, alive: Vec<bool>, heap: BinaryHeap<Reverse<(u32, u32)>> }
impl MS { fn new() -> Self { MS { next: Vec::new(), prev: Vec::new(), alive: Vec::new(), heap: BinaryHeap::new() } } }

impl Bpe {
    fn load(path: &str) -> Self {
        let j: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let m = &j["model"];
        let mut vocab: Map<Vec<u8>, u32> = newmap();
        for (k, v) in m["vocab"].as_object().unwrap() { vocab.insert(k.as_bytes().to_vec(), v.as_u64().unwrap() as u32); }
        let b2u = bytes_to_unicode();
        let mut char2byte: Map<char, u8> = newmap();
        for b in 0..256 { char2byte.insert(b2u[b], b as u8); }
        let mut byte_id = [0u32; 256];
        for b in 0..256 { let s = b2u[b].to_string(); byte_id[b] = *vocab.get(s.as_bytes()).unwrap_or(&0); }
        let mut ranks: Map<(u32, u32), (u32, u32)> = newmap();
        for (rank, pair) in m["merges"].as_array().unwrap().iter().enumerate() {
            let (a, b) = match pair { Value::Array(xs) => (xs[0].as_str().unwrap(), xs[1].as_str().unwrap()),
                Value::String(s) => { let mut it = s.splitn(2, ' '); (it.next().unwrap(), it.next().unwrap()) }, _ => continue };
            let mut ab = a.as_bytes().to_vec(); ab.extend_from_slice(b.as_bytes());
            if let (Some(&ia), Some(&ib), Some(&iab)) = (vocab.get(a.as_bytes()), vocab.get(b.as_bytes()), vocab.get(&ab)) { ranks.insert((ia, ib), (rank as u32, iab)); }
        }
        let mut vocab_raw: Map<Vec<u8>, u32> = newmap();
        for (k, &id) in &vocab { let s = match std::str::from_utf8(k) { Ok(s) => s, Err(_) => continue }; let mut raw = Vec::with_capacity(s.len()); let mut ok = true;
            for ch in s.chars() { if let Some(&b) = char2byte.get(&ch) { raw.push(b); } else { ok = false; break; } } if ok { vocab_raw.insert(raw, id); } }
        // SentencePiece-style vocab (gemma/mistral): byte-level inversion fails -> keep as-stored keys (same footprint shape)
        if vocab_raw.len() * 10 < vocab.len() * 9 { vocab_raw.clear(); for (k, &id) in &vocab { vocab_raw.insert(k.clone(), id); } }
        Bpe { vocab_raw, ranks, byte_id }
    }
    #[inline]
    fn piece(&self, p: &[u8], out: &mut Vec<u32>, syms: &mut Vec<u32>, ms: &mut MS) {
        if p.is_empty() { return; }
        if let Some(&id) = self.vocab_raw.get(p) { out.push(id); return; }
        syms.clear(); for &b in p { syms.push(self.byte_id[b as usize]); }
        if syms.len() == 1 { out.push(syms[0]); return; }
        let n = syms.len();
        ms.next.clear(); ms.prev.clear(); ms.alive.clear(); ms.heap.clear();
        for i in 0..n { ms.next.push(i as i32 + 1); ms.prev.push(i as i32 - 1); ms.alive.push(true); }
        for i in 0..n - 1 { if let Some(&(r, _)) = self.ranks.get(&(syms[i], syms[i + 1])) { ms.heap.push(Reverse((r, i as u32))); } }
        while let Some(Reverse((r, pos))) = ms.heap.pop() {
            let i = pos as usize; if !ms.alive[i] { continue; }
            let j = ms.next[i]; if j < 0 || j as usize >= n || !ms.alive[j as usize] { continue; } let j = j as usize;
            if let Some(&(rr, mm)) = self.ranks.get(&(syms[i], syms[j])) { if rr == r {
                syms[i] = mm; ms.alive[j] = false; let nj = ms.next[j]; ms.next[i] = nj;
                if nj >= 0 && (nj as usize) < n { ms.prev[nj as usize] = i as i32; }
                let pi = ms.prev[i];
                if pi >= 0 { if let Some(&(r2, _)) = self.ranks.get(&(syms[pi as usize], syms[i])) { ms.heap.push(Reverse((r2, pi as u32))); } }
                if nj >= 0 && (nj as usize) < n { if let Some(&(r2, _)) = self.ranks.get(&(syms[i], syms[nj as usize])) { ms.heap.push(Reverse((r2, i as u32))); } }
            } }
        }
        let mut i = 0usize; loop { out.push(syms[i]); let nx = ms.next[i]; if nx < 0 || (nx as usize) >= n { break; } i = nx as usize; }
    }
}

// scalar fallback for the DFA's None cases (ASCII whitespace rules 5/6; non-ASCII best-effort) — no onig.
#[inline]
fn fallback(t: &[u8], i: usize, end: usize) -> usize {
    let b = t[i];
    if b >= 0x80 { return (i + match b { 0xf0..=0xf4 => 4, 0xe0..=0xef => 3, 0xc0..=0xdf => 2, _ => 1 }).min(end); }
    let mut m = i; while m < end && t[m] < 0x80 && wsa(t[m]) { m += 1; }
    if m == i { return i + 1; }
    let mut last_nl = None; for k in i..m { if nl(t[k]) { last_nl = Some(k); } }
    if let Some(ln) = last_nl { return ln + 1; }            // rule 5
    if m == end { m } else if m - 1 > i { m - 1 } else { m } // rule 6
}

struct Enc { bpe: Bpe, cl_l: Cls, cl_p: Cls, cl_s: Cls }
impl Enc {
    fn encode(&self, t: &[u8], out: &mut Vec<u32>, syms: &mut Vec<u32>, ms: &mut MS) {
        out.clear();
        let mut i = 0;
        while i < t.len() {
            if let Some(j) = fast_token(t, i, t.len(), &self.cl_l, &self.cl_p, &self.cl_s) { self.bpe.piece(&t[i..j], out, syms, ms); i = j; }
            else { let j = fallback(t, i, t.len()); self.bpe.piece(&t[i..j], out, syms, ms); i = j; }
        }
    }
}
// DFA token (rules 1-4 + pure-space 5/6), None -> scalar fallback. (copy of pocenc fast_token)
#[inline]
fn fast_token(t: &[u8], i: usize, end: usize, cl_l: &Cls, cl_p: &Cls, cl_s: &Cls) -> Option<usize> {
    let b = t[i]; if b >= 0x80 { return None; }
    if b == b'\'' && i + 1 < end && t[i + 1] < 0x80 { let c = t[i + 1].to_ascii_lowercase();
        match c { b's'|b't'|b'm'|b'd' => return Some(i + 2),
            b'r'|b'v'|b'l' => { if i + 2 < end && t[i + 2] < 0x80 { let c2 = t[i + 2].to_ascii_lowercase();
                if (c==b'r'&&c2==b'e')||(c==b'v'&&c2==b'e')||(c==b'l'&&c2==b'l') { return Some(i + 3); } } }, _ => {} } }
    if !nl(b) && !letter(b) && !digit(b) { let k = first_not(t, i + 1, end, cl_l); if k > i + 1 { if k < end && t[k] >= 0x80 { return None; } return Some(k); } }
    if letter(b) { let k = first_not(t, i, end, cl_l); if k < end && t[k] >= 0x80 { return None; } return Some(k); }
    if digit(b) { let (mut k, mut c) = (i + 1, 1); while k < end && c < 3 { let x = t[k]; if x >= 0x80 { return None; } if digit(x) { k += 1; c += 1; } else { break; } } return Some(k); }
    { let sp = if b == b' ' { i + 1 } else { i }; let k = first_not(t, sp, end, cl_p);
        if k > sp { if k < end && t[k] >= 0x80 { return None; } let mut e = k; while e < end { let x = t[e]; if x >= 0x80 { break; } if nl(x) { e += 1; } else { break; } } return Some(e); } }
    if b == b' ' { let m = first_not(t, i, end, cl_s); if m == end { return Some(m); } let y = t[m];
        if y < 0x80 && (letter(y) || (!wsa(y) && !digit(y))) { return Some(m - 1); } }
    None
}

fn main() {
    let base = live();
    let path = std::env::args().nth(1).unwrap_or_else(|| format!("{}/../../tokenizers/data/llama-3-tokenizer.json", env!("CARGO_MANIFEST_DIR")));
    let bpe = Bpe::load(&path);
    let cl_l = build_cls(letter);
    let cl_p = build_cls(|c| c < 0x80 && !wsa(c) && !letter(c) && !digit(c));
    let cl_s = build_cls(|c| c == b' ');
    let enc = Enc { bpe, cl_l, cl_p, cl_s };
    let loaded = live() - base;
    if std::env::var("MEM_ONLY").is_ok() { println!("pocmin: loaded structures = {:.2} MB", loaded as f64 / 1e6); black_box(&enc); return; }
    let big = std::fs::read_to_string(format!("{}/../../tokenizers/data/big.txt", env!("CARGO_MANIFEST_DIR"))).unwrap();
    let (mut o, mut s, mut ms) = (Vec::new(), Vec::new(), MS::new());
    enc.encode(big.as_bytes(), &mut o, &mut s, &mut ms);
    let ntok = o.len();
    for _ in 0..3 { enc.encode(black_box(big.as_bytes()), &mut o, &mut s, &mut ms); }
    let t = Instant::now(); for _ in 0..10 { enc.encode(black_box(big.as_bytes()), &mut o, &mut s, &mut ms); black_box(&o); }
    let nsb = t.elapsed().as_nanos() as f64 / (10 * big.len()) as f64;
    println!("pocmin: loaded structures = {:.2} MB, big.txt -> {} tokens, {:.2} ns/byte ({:.0} MB/s)", loaded as f64 / 1e6, ntok, nsb, 1000.0 / nsb);
}
