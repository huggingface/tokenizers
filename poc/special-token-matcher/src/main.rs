// FINAL bench: VocabStore-based AddedVocabulary  vs  the old HF design, on real prompts.
//
//   NEW: one VocabStore (MPHF + single byte slab) serves token_to_id, id_to_token AND special
//        matching. The matcher adds ONLY per-bucket {prefix, lengths} -- no extra strings,
//        no automaton. Matching = memchr/table -> prefix reject -> probe distinct lengths,
//        each a VocabStore lookup (mphf-1).
//   OLD: HashMap<String,u32> (fwd) + HashMap<u32,String> (rev) + daachorse split_trie (match).
//        Strings owned ~3x, plus a heavy double-array automaton.
//
//   PROMPTS=../prompts.json cargo run --release --offline

use ahash::RandomState;
use daachorse::{DoubleArrayAhoCorasick, DoubleArrayAhoCorasickBuilder, MatchKind};
use memchr::memchr;
use ptr_hash::bucket_fn::Linear;
use ptr_hash::{PtrHash, PtrHashParams};
use serde::Deserialize;
use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::HashMap;
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::time::Instant;

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

#[derive(Deserialize)]
struct Case { name: String, kind: String, specials: Vec<String>, prompt: String }

// ---- VocabStore (mirrors tk-encode): MPHF + single slab; serves fwd, rev, and membership ----
type Mphf = PtrHash<u64, Linear>;
const SEEDS: [u64; 4] = [0x243F6A8885A308D3, 0x13198A2E03707344, 0xA4093822299F31D0, 0x082EFA98EC4E6C89];
struct VocabStore { mphf: Mphf, hasher: RandomState, slab: Vec<u8>, ent: Vec<(u32, u16, u32)>, id_to_slot: Vec<u32> }
impl VocabStore {
    fn build(items: &[(&[u8], u32)]) -> Self {
        let hasher = RandomState::with_seeds(SEEDS[0], SEEDS[1], SEEDS[2], SEEDS[3]);
        let keys: Vec<u64> = items.iter().map(|(s, _)| hasher.hash_one(*s)).collect();
        let mut p = PtrHashParams::default_fast();
        p.single_part = true;
        let mphf = Mphf::new(&keys, p);
        let maxid = items.iter().map(|(_, id)| *id).max().unwrap_or(0);
        let (mut ent, mut slab, mut id_to_slot) = (vec![(0u32, 0u16, 0u32); items.len()], Vec::new(), vec![u32::MAX; maxid as usize + 1]);
        for (s, id) in items {
            let slot = mphf.index_single_part(&hasher.hash_one(*s));
            ent[slot] = (slab.len() as u32, s.len() as u16, *id);
            id_to_slot[*id as usize] = slot as u32;
            slab.extend_from_slice(s);
        }
        VocabStore { mphf, hasher, slab, ent, id_to_slot }
    }
    #[inline]
    fn token_to_id(&self, q: &[u8]) -> Option<u32> {
        if self.ent.is_empty() { return None; }
        let (off, len, id) = self.ent[self.mphf.index_single_part(&self.hasher.hash_one(q))];
        if len as usize == q.len() && self.slab[off as usize..off as usize + len as usize] == *q { Some(id) } else { None }
    }
    #[inline]
    fn id_to_token(&self, id: u32) -> Option<&[u8]> {
        let slot = *self.id_to_slot.get(id as usize)?;
        if slot == u32::MAX { return None; }
        let (off, len, _) = self.ent[slot as usize];
        Some(&self.slab[off as usize..off as usize + len as usize])
    }
}

// matcher metadata: ONLY prefix + distinct lengths per bucket (no strings)
struct Bucket { prefix: Box<[u8]>, lengths: Box<[u16]> }
struct Buckets { fb: [u8; 256], list: Vec<Bucket>, single: Option<u8> }
impl Buckets {
    fn build(items: &[&[u8]]) -> Self {
        let mut sorted = items.to_vec();
        sorted.sort_by(|a, b| a[0].cmp(&b[0]).then(b.len().cmp(&a.len())));
        let mut fb = [u8::MAX; 256];
        let mut list = Vec::new();
        let mut i = 0;
        while i < sorted.len() {
            let fbyte = sorted[i][0];
            let mut j = i;
            while j < sorted.len() && sorted[j][0] == fbyte { j += 1; }
            let group = &sorted[i..j];
            let mut prefix = group[0].to_vec();
            let mut lengths: Vec<u16> = Vec::new();
            for g in group {
                let c = prefix.iter().zip(*g).take_while(|(a, b)| a == b).count().max(1);
                prefix.truncate(c);
                let l = g.len() as u16;
                if !lengths.contains(&l) { lengths.push(l); }
            }
            lengths.sort_unstable_by(|a, b| b.cmp(a));
            fb[fbyte as usize] = list.len() as u8;
            list.push(Bucket { prefix: prefix.into(), lengths: lengths.into() });
            i = j;
        }
        let single = if list.len() == 1 { Some(list[0].prefix[0]) } else { None };
        Buckets { fb, list, single }
    }
    // mphf-1 scan: reuses the VocabStore for membership.
    fn scan(&self, t: &[u8], vs: &VocabStore) -> u64 {
        let (mut hits, mut s) = (0u64, 0usize);
        while let Some(ms) = match self.single {
            Some(b) => memchr(b, &t[s..]).map(|r| s + r),
            None => t[s..].iter().position(|&c| self.fb[c as usize] != u8::MAX).map(|r| s + r),
        } {
            let rem = &t[ms..];
            let b = &self.list[self.fb[rem[0] as usize] as usize];
            if rem.len() < b.prefix.len() || rem[..b.prefix.len()] != b.prefix[..] { s = ms + 1; continue; }
            let mut step = 1;
            for &l in b.lengths.iter() {
                let l = l as usize;
                if l <= rem.len() && vs.token_to_id(&rem[..l]).is_some() { hits += 1; step = l; break; }
            }
            s = ms + step;
        }
        hits
    }
}

// Faithful IREE special_tokens.c design:
//  - Level-1: first_byte_to_bucket[256]; Level-2: per-bucket prefix CAPPED AT 4 BYTES.
//  - Tokens in a single contiguous slab (B-string style), sorted by length DESC within a bucket
//    so the first linear-scan match is the longest. Reject hot path = 1 fetch; match = bucket scan.
struct IreeBucket { prefix: [u8; 4], plen: usize, start: u32, end: u32 } // [start,end) into offs
struct IreeScan { fb: [u8; 256], buckets: Vec<IreeBucket>, offs: Vec<(u32, u16)>, slab: Vec<u8>, single: Option<u8> }
impl IreeScan {
    fn build(items: &[&[u8]]) -> Self {
        let mut sorted = items.to_vec();
        // bucket by first byte, then length DESC within bucket (IREE ordering)
        sorted.sort_by(|a, b| a[0].cmp(&b[0]).then(b.len().cmp(&a.len())));
        let mut fb = [u8::MAX; 256];
        let (mut buckets, mut offs, mut slab) = (Vec::new(), Vec::new(), Vec::new());
        let mut i = 0;
        while i < sorted.len() {
            let fbyte = sorted[i][0];
            let mut j = i;
            while j < sorted.len() && sorted[j][0] == fbyte { j += 1; }
            let group = &sorted[i..j];
            // common prefix, capped at 4 bytes (IREE prefix_length is 1..=4)
            let mut lcp = group[0].len();
            for g in group { lcp = lcp.min(group[0].iter().zip(*g).take_while(|(a, b)| a == b).count()); }
            let plen = lcp.clamp(1, 4);
            let mut prefix = [0u8; 4];
            prefix[..plen].copy_from_slice(&group[0][..plen]);
            let start = offs.len() as u32;
            for g in group { offs.push((slab.len() as u32, g.len() as u16)); slab.extend_from_slice(g); }
            fb[fbyte as usize] = buckets.len() as u8;
            buckets.push(IreeBucket { prefix, plen, start, end: offs.len() as u32 });
            i = j;
        }
        let single = if buckets.len() == 1 { Some(sorted[0][0]) } else { None };
        IreeScan { fb, buckets, offs, slab, single }
    }
    fn scan(&self, t: &[u8]) -> u64 {
        let (mut hits, mut s) = (0u64, 0usize);
        while let Some(ms) = match self.single {
            Some(b) => memchr(b, &t[s..]).map(|r| s + r),
            None => t[s..].iter().position(|&c| self.fb[c as usize] != u8::MAX).map(|r| s + r),
        } {
            let rem = &t[ms..];
            let b = &self.buckets[self.fb[rem[0] as usize] as usize];
            if rem.len() < b.plen || rem[..b.plen] != b.prefix[..b.plen] { s = ms + 1; continue; }
            let mut step = 1;
            for k in b.start..b.end { // contiguous slab, longest-first
                let (off, len) = self.offs[k as usize];
                let (off, len) = (off as usize, len as usize);
                if len <= rem.len() && self.slab[off..off + len] == rem[..len] { hits += 1; step = len; break; }
            }
            s = ms + step;
        }
        hits
    }
}

const ITERS: usize = 4000;
fn ns_byte(t: usize, f: &dyn Fn() -> u64) -> f64 {
    for _ in 0..3 { black_box(f()); }
    let s = Instant::now();
    let mut a = 0u64;
    for _ in 0..ITERS { a += f(); }
    black_box(a);
    s.elapsed().as_nanos() as f64 / (ITERS * t) as f64
}
fn ns_op(n: usize, f: &dyn Fn() -> u64) -> f64 {
    for _ in 0..3 { black_box(f()); }
    let s = Instant::now();
    let mut a = 0u64;
    for _ in 0..ITERS { a += f(); }
    black_box(a);
    s.elapsed().as_nanos() as f64 / (ITERS * n) as f64
}

fn main() {
    let path = std::env::var("PROMPTS").unwrap_or_else(|_| format!("{}/prompts.json", env!("CARGO_MANIFEST_DIR")));
    let cases: Vec<Case> = serde_json::from_slice(&std::fs::read(&path).expect("prompts.json")).unwrap();

    // Warm up one-time global init (ptr_hash/rayon, daachorse) so the first case's memory
    // delta isn't polluted by it.
    {
        let dummy: Vec<&[u8]> = vec![b"<|a|>", b"<|b|>", b"[x]"];
        let pairs: Vec<(&[u8], u32)> = dummy.iter().enumerate().map(|(i, s)| (*s, i as u32)).collect();
        let vs = VocabStore::build(&pairs);
        let bk = Buckets::build(&dummy);
        let pats: Vec<Vec<u8>> = dummy.iter().map(|b| b.to_vec()).collect();
        let pma: DoubleArrayAhoCorasick<u32> = DoubleArrayAhoCorasickBuilder::new().match_kind(MatchKind::LeftmostLongest).build(&pats).unwrap();
        black_box((bk.scan(b"<|a|> hi [x]", &vs), pma.leftmost_find_iter(b"<|a|>").count(), vs.token_to_id(b"<|a|>")));
    }

    println!("SPEED (ns/op for lookups, ns/byte for match-scan on the prompt)");
    println!("{:<18} {:>5} {:>4} | {:>11} | {:>11} | {:>20} | {:>11}", "case", "spec", "hits", "token_to_id", "id_to_token", "match-scan (ns/byte)", "matcher mem");
    println!("{:<18} {:>5} {:>4} | {:>5} {:>5} | {:>5} {:>5} | {:>6} {:>6} {:>6} | {:>5} {:>5}", "", "", "", "MPHF", "Hmap", "MPHF", "Hmap", "MPHF", "IREE", "daac", "MPHF", "IREE");
    println!("{}", "-".repeat(104));

    let mut tot_new = 0usize;
    let mut tot_old = 0usize;
    let mut tot_str = 0usize;
    let mut memrows: Vec<(String, usize, usize, usize, usize, usize, usize, usize)> = Vec::new();

    for c in &cases {
        let t = c.prompt.as_bytes();
        let mut items: Vec<&[u8]> = c.specials.iter().map(|s| s.as_bytes()).filter(|b| !b.is_empty()).collect();
        items.sort_unstable();
        items.dedup();
        let n = items.len();
        let pairs: Vec<(&[u8], u32)> = items.iter().enumerate().map(|(i, s)| (*s, i as u32)).collect();
        let str_bytes: usize = items.iter().map(|s| s.len()).sum();

        // ---- NEW: VocabStore + bucket metadata ----
        let v0 = live();
        let vs = VocabStore::build(&pairs);
        let vs_mem = live() - v0;
        let b0 = live();
        let bk = Buckets::build(&items);
        let bk_mem = live() - b0;
        let new_mem = vs_mem + bk_mem;

        // ---- OLD: fwd map + rev map + daachorse ----
        let f0 = live();
        let fwd: HashMap<Vec<u8>, u32> = items.iter().enumerate().map(|(i, s)| (s.to_vec(), i as u32)).collect();
        let fwd_mem = live() - f0;
        let r0 = live();
        let rev: HashMap<u32, Vec<u8>> = items.iter().enumerate().map(|(i, s)| (i as u32, s.to_vec())).collect();
        let rev_mem = live() - r0;
        let pats: Vec<Vec<u8>> = items.iter().map(|b| b.to_vec()).collect();
        let a0 = live();
        let pma: DoubleArrayAhoCorasick<u32> = DoubleArrayAhoCorasickBuilder::new().match_kind(MatchKind::LeftmostLongest).build(&pats).unwrap();
        let ac_mem = live() - a0;
        let old_mem = fwd_mem + rev_mem + ac_mem;

        // ---- IREE-style scan (first-byte + prefix reject + longest-first linear compare) ----
        let i0 = live();
        let ir = IreeScan::build(&items);
        let iree_mem = live() - i0;

        // ---- correctness ----
        let h_new = bk.scan(t, &vs);
        let h_old = pma.leftmost_find_iter(t).count() as u64;
        let h_iree = ir.scan(t);
        let ok = h_new == h_old && h_iree == h_old;

        // ---- speed: token_to_id ----
        let tti_new = ns_op(n, &|| { let mut s = 0u64; for it in &items { s += vs.token_to_id(black_box(it)).unwrap_or(0) as u64; } s });
        let tti_old = ns_op(n, &|| { let mut s = 0u64; for it in &items { s += *fwd.get(black_box(*it)).unwrap_or(&0) as u64; } s });
        // ---- speed: id_to_token ----
        let itt_new = ns_op(n, &|| { let mut s = 0u64; for id in 0..n as u32 { s += vs.id_to_token(black_box(id)).map_or(0, |b| b.len() as u64); } s });
        let itt_old = ns_op(n, &|| { let mut s = 0u64; for id in 0..n as u32 { s += rev.get(black_box(&id)).map_or(0, |b| b.len() as u64); } s });
        // ---- speed: match-scan (ns/byte): MPHF length-probe vs IREE linear scan vs daachorse AC ----
        let m_new = ns_byte(t.len(), &|| bk.scan(black_box(t), &vs));
        let m_iree = ns_byte(t.len(), &|| ir.scan(black_box(t)));
        let m_old = ns_byte(t.len(), &|| pma.leftmost_find_iter(black_box(t)).count() as u64);

        println!("{:<18} {:>5} {:>4} | {:>5.1} {:>5.1} | {:>5.1} {:>5.1} | {:>6.2} {:>6.2} {:>6.2} | {:>5} {:>5} {}",
            format!("{}/{}", c.name, c.kind), n, h_new,
            tti_new, tti_old, itt_new, itt_old, m_new, m_iree, m_old, bk_mem, iree_mem, if ok { "" } else { "MISMATCH!" });

        tot_new += new_mem; tot_old += old_mem; tot_str += str_bytes;
        memrows.push((format!("{}/{}", c.name, c.kind), str_bytes, vs_mem, bk_mem, new_mem, fwd_mem + rev_mem, ac_mem, old_mem));
    }

    println!("\nMEMORY (real heap, bytes)");
    println!("{:<20} {:>7} | {:>8} {:>6} {:>8} | {:>10} {:>8} {:>8} | {:>6}", "case", "strbytes", "vstore", "bkts", "NEW-tot", "2xHashMap", "daac", "OLD-tot", "x");
    println!("{}", "-".repeat(104));
    for (name, sb, vsm, bkm, nm, hm, acm, om) in &memrows {
        println!("{:<20} {:>7} | {:>8} {:>6} {:>8} | {:>10} {:>8} {:>8} | {:>5.1}x", name, sb, vsm, bkm, nm, hm, acm, om, *om as f64 / *nm as f64);
    }
    println!("{}", "-".repeat(104));
    println!("{:<20} {:>7} | {:>8} {:>6} {:>8} | {:>10} {:>8} {:>8} | {:>5.1}x",
        "TOTAL", tot_str, "", "", tot_new, "", "", tot_old, tot_old as f64 / tot_new as f64);
    println!("\nNEW matcher extra over VocabStore = 'bkts' col (prefix+lengths only). OLD matcher = full daachorse trie.");
}
