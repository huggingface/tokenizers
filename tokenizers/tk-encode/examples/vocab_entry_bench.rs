//! Does `WordCache`'s packed key pay off in `BucketVocabStore`? All variants in
//! **one binary**, because cross-binary timings in this workspace swing ±30% on
//! code layout alone.
//!
//! `get_bytes` is three dependent memory accesses: the MPHF's pivots, then
//! `entries[slot]`, then the token's bytes in the slab, followed by a `memcmp`. The
//! question is whether the two tricks that made `WordCache`'s key cheap help here:
//!
//! 1. **Hash the packed key** instead of the token's bytes — `WordCache::slot_key`.
//!    Pure ALU saving, so it only shows if the lookup is not memory-bound.
//! 2. **Put the token in the entry** — `WordCache`'s slot carries the word itself
//!    for anything up to 15 bytes, so confirming a hit is one register compare.
//!    That removes the slab access and the `memcmp` outright, at 32 bytes per slot
//!    instead of 12.
//!
//! The word stream is the real one: the model's own normalizer and pre-tokenizer
//! over a real corpus, so hit rate, token-length mix and query skew are whatever
//! the model and the language actually produce — a uniform sweep over the vocab
//! would flatter (2) by making every lookup miss cache. Words land in one
//! contiguous buffer, so each has the 16-byte window `Split::head` provides.
//!
//! The shipped `BucketVocabStore` is measured alongside as a fidelity control on
//! the copies. Every variant must return identical ids: the checksum is asserted,
//! not printed, because an earlier draft of this comparison silently benched a
//! hit-heavy variant against an all-miss one.
//!
//! ```text
//! cargo run --release --example vocab_entry_bench
//! cargo run --release --example vocab_entry_bench -- gpt2 eng_Latn
//! ```

use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::Instant;

use ahash::RandomState;
use ptr_hash::{FastPtrHash, PtrHashParams, hash::NoHash};
use serde_json::Value;
use tk_encode::vocab::bucket_vocab_store::BucketVocabStore;
use tk_encode::{
    NormalizedString, Normalizer, OffsetReferential, OffsetType, PreTokenizedString, PreTokenizer,
    Tokenizer,
};

const DATA_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../data");
const MODELS: [&str; 2] = ["gpt2", "llama-3-tokenizer"];
const CORPORA: [&str; 3] = ["eng_Latn", "cmn_Hani", "code_mixed"];
/// Real encode sees ~10 kB inputs; pre-tokenization is per input.
const CHUNK_BYTES: usize = 10 * 1024;
const MAX_WORDS: usize = 300_000;
/// Reported number is the **minimum** over all reps: interference from the rest of
/// the machine can only add time, so the fastest pass is closest to the cost of the
/// code itself.
const REPS: usize = 7;
const SEEDS: (u64, u64, u64, u64) = (0x5eed_0001, 0x5eed_0002, 0x5eed_0003, 0x5eed_0004);

// ---------------------------------------------------------------------------
// The entry layouts under test
// ---------------------------------------------------------------------------

type Mphf = FastPtrHash<NoHash, u64>;

/// What ships: coordinates into the byte slab.
#[derive(Clone, Copy, Default)]
#[repr(C)]
struct Thin {
    start: u32,
    len: u16,
    id: u32,
}

/// `WordCache`'s slot: the token itself when it fits in 15 bytes, otherwise its
/// hash with [`LONG_TAG`] set and the slab coordinates beside it.
#[derive(Clone, Copy, Default)]
#[repr(C)]
struct Fat {
    key: u128,
    id: u32,
    start: u32,
    len: u16,
    _pad: u16,
}

/// [`Fat`] with the key as bytes, which drops `u128`'s 16-byte alignment and with
/// it four bytes of tail padding per slot.
#[derive(Clone, Copy, Default)]
#[repr(C)]
struct Slim {
    key: [u8; 16],
    id: u32,
    start: u32,
    len: u16,
    _pad: u16,
}

const LONG_TAG: u128 = 1 << 127;

/// One lookup shape: the query, and the 16-byte window when the caller has one.
type Lookup = fn(&Store, &[u8], Option<&[u8; 16]>) -> Option<u32>;

/// Packing from a bare slice: the copy length is only known at run time, so this is
/// a call into `memcpy`. Used at build time and wherever a query has no window.
fn pack_owned(token: &[u8]) -> Option<u128> {
    let len = token.len();
    if len == 0 || len > 15 {
        return None;
    }
    let mut lanes = [0u8; 16];
    lanes[..len].copy_from_slice(token);
    Some(u128::from_le_bytes(lanes) | ((len as u128) << 120))
}

/// Packing from a window the caller already holds — one load and a mask.
#[inline(always)]
fn pack_window(len: usize, head: Option<&[u8; 16]>) -> Option<u128> {
    if len == 0 || len > 15 {
        return None;
    }
    let lanes = u128::from_le_bytes(*head?);
    Some((lanes & (u128::MAX >> (8 * (16 - len)))) | ((len as u128) << 120))
}

#[inline(always)]
fn key_of(h: &RandomState, token: &[u8]) -> u64 {
    match pack_owned(token) {
        Some(packed) => h.hash_one(packed),
        None => h.hash_one(token),
    }
}

struct Store {
    h: RandomState,
    /// Keyed on the tokens' bytes, as the shipped store is.
    mphf_bytes: Mphf,
    /// Keyed on the packed tokens — trick 1.
    mphf_packed: Mphf,
    slab: Vec<u8>,
    thin_bytes: Vec<Thin>,
    thin_packed: Vec<Thin>,
    fat: Vec<Fat>,
    slim: Vec<Slim>,
}

impl Store {
    fn build(tokens: &[(Vec<u8>, u32)]) -> Self {
        let h = RandomState::with_seeds(SEEDS.0, SEEDS.1, SEEDS.2, SEEDS.3);
        let params = PtrHashParams::default_fast();
        let mphf_bytes = Mphf::new(
            &tokens
                .iter()
                .map(|(t, _)| h.hash_one(t.as_slice()))
                .collect::<Vec<_>>(),
            params,
        );
        let mphf_packed = Mphf::new(
            &tokens
                .iter()
                .map(|(t, _)| key_of(&h, t))
                .collect::<Vec<_>>(),
            params,
        );

        let mut slab = Vec::new();
        let mut thin_bytes = vec![Thin::default(); mphf_bytes.max_index()];
        let mut thin_packed = vec![Thin::default(); mphf_packed.max_index()];
        let mut fat = vec![Fat::default(); mphf_packed.max_index()];
        let mut slim = vec![Slim::default(); mphf_packed.max_index()];
        for (token, id) in tokens {
            let start = slab.len() as u32;
            slab.extend_from_slice(token);
            let thin = Thin {
                start,
                len: token.len() as u16,
                id: *id,
            };
            thin_bytes[mphf_bytes.index(&h.hash_one(token.as_slice()))] = thin;
            let slot = mphf_packed.index(&key_of(&h, token));
            thin_packed[slot] = thin;
            let key =
                pack_owned(token).unwrap_or((h.hash_one(token.as_slice()) as u128) | LONG_TAG);
            fat[slot] = Fat {
                key,
                id: *id,
                start,
                len: token.len() as u16,
                _pad: 0,
            };
            slim[slot] = Slim {
                key: key.to_le_bytes(),
                id: *id,
                start,
                len: token.len() as u16,
                _pad: 0,
            };
        }
        Self {
            h,
            mphf_bytes,
            mphf_packed,
            slab,
            thin_bytes,
            thin_packed,
            fat,
            slim,
        }
    }

    fn entries_bytes(&self) -> (usize, usize, usize) {
        (
            size_of::<Thin>() * self.thin_bytes.len(),
            size_of::<Fat>() * self.fat.len(),
            size_of::<Slim>() * self.slim.len(),
        )
    }
}

/// Today: hash the bytes, confirm against the slab.
#[inline(never)]
fn v_today(s: &Store, q: &[u8], _head: Option<&[u8; 16]>) -> Option<u32> {
    let e = s.thin_bytes[s.mphf_bytes.index(&s.h.hash_one(q))];
    let (start, len) = (e.start as usize, e.len as usize);
    (len == q.len() && s.slab[start..start + len] == *q).then_some(e.id)
}

/// Trick 1 only: hash the packed key, still confirm against the slab.
#[inline(never)]
fn v_packed_hash(s: &Store, q: &[u8], head: Option<&[u8; 16]>) -> Option<u32> {
    let hash = match pack_window(q.len(), head) {
        Some(packed) => s.h.hash_one(packed),
        None => key_of(&s.h, q),
    };
    let e = s.thin_packed[s.mphf_packed.index(&hash)];
    let (start, len) = (e.start as usize, e.len as usize);
    (len == q.len() && s.slab[start..start + len] == *q).then_some(e.id)
}

/// Tricks 1 and 2: the token lives in the entry, so a short token never touches
/// the slab and the comparison is one register-wide equality.
#[inline(never)]
fn v_fat(s: &Store, q: &[u8], head: Option<&[u8; 16]>) -> Option<u32> {
    if let Some(packed) = pack_window(q.len(), head) {
        let e = s.fat[s.mphf_packed.index(&s.h.hash_one(packed))];
        return (e.key == packed).then_some(e.id);
    }
    let hash = key_of(&s.h, q);
    let e = s.fat[s.mphf_packed.index(&hash)];
    if let Some(packed) = pack_owned(q) {
        return (e.key == packed).then_some(e.id);
    }
    let (start, len) = (e.start as usize, e.len as usize);
    (e.key == (hash as u128) | LONG_TAG && len == q.len() && s.slab[start..start + len] == *q)
        .then_some(e.id)
}

/// [`v_fat`] with no window from the caller, so the query is packed with the
/// run-time-length copy. This is what a `get_bytes(&[u8])` that keeps its current
/// signature would cost.
#[inline(never)]
fn v_fat_owned(s: &Store, q: &[u8], _head: Option<&[u8; 16]>) -> Option<u32> {
    // Pack once and hash what was packed, so this pays for exactly one copy.
    if let Some(packed) = pack_owned(q) {
        let e = s.fat[s.mphf_packed.index(&s.h.hash_one(packed))];
        return (e.key == packed).then_some(e.id);
    }
    let hash = s.h.hash_one(q);
    let e = s.fat[s.mphf_packed.index(&hash)];
    let (start, len) = (e.start as usize, e.len as usize);
    (e.key == (hash as u128) | LONG_TAG && len == q.len() && s.slab[start..start + len] == *q)
        .then_some(e.id)
}

/// As [`v_fat`], on the 28-byte entry.
#[inline(never)]
fn v_slim(s: &Store, q: &[u8], head: Option<&[u8; 16]>) -> Option<u32> {
    if let Some(packed) = pack_window(q.len(), head) {
        let e = s.slim[s.mphf_packed.index(&s.h.hash_one(packed))];
        return (u128::from_le_bytes(e.key) == packed).then_some(e.id);
    }
    let hash = key_of(&s.h, q);
    let e = s.slim[s.mphf_packed.index(&hash)];
    let key = u128::from_le_bytes(e.key);
    if let Some(packed) = pack_owned(q) {
        return (key == packed).then_some(e.id);
    }
    let (start, len) = (e.start as usize, e.len as usize);
    (key == (hash as u128) | LONG_TAG && len == q.len() && s.slab[start..start + len] == *q)
        .then_some(e.id)
}

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------

fn vocab_of(model_json: &Path) -> Vec<(Vec<u8>, u32)> {
    let raw: Value = serde_json::from_str(&std::fs::read_to_string(model_json).unwrap()).unwrap();
    let map = raw["model"]["vocab"].as_object().expect("model.vocab");
    // The tokens as the store holds them before byte-level un-mapping, which is the
    // same alphabet the byte-level pre-tokenizer emits below — so the two pair up.
    map.iter()
        .map(|(token, id)| (token.as_bytes().to_vec(), id.as_u64().unwrap() as u32))
        .collect()
}

/// The model's own normalizer and pre-tokenizer over `corpus`, flattened into one
/// buffer so every word has a 16-byte window after it.
fn word_stream(tok: &Tokenizer, corpus: &str) -> (Vec<u8>, Vec<(usize, usize)>) {
    let mut buf = Vec::new();
    let mut spans = Vec::new();
    for chunk in chunks(corpus, CHUNK_BYTES) {
        if spans.len() >= MAX_WORDS {
            break;
        }
        let mut normalized = NormalizedString::from(chunk);
        if let Some(n) = tok.get_normalizer() {
            n.normalize(&mut normalized).unwrap();
        }
        let mut pretokenized = PreTokenizedString::from(normalized);
        if let Some(pt) = tok.get_pre_tokenizer() {
            pt.pre_tokenize(&mut pretokenized).unwrap();
        }
        for (word, _, _) in pretokenized.get_splits(OffsetReferential::Normalized, OffsetType::Byte)
        {
            if word.is_empty() {
                continue;
            }
            spans.push((buf.len(), word.len()));
            buf.extend_from_slice(word.as_bytes());
        }
    }
    // Slack so the final word still has a window, as a chunk's interior words do.
    buf.extend_from_slice(&[0u8; 16]);
    (buf, spans)
}

fn chunks(text: &str, size: usize) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    while start < text.len() {
        let mut end = (start + size).min(text.len());
        while end < text.len() && !text.is_char_boundary(end) {
            end += 1;
        }
        out.push(&text[start..end]);
        start = end;
    }
    out
}

// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let models: Vec<&str> = if args.is_empty() {
        MODELS.to_vec()
    } else {
        MODELS
            .into_iter()
            .filter(|m| args.iter().any(|a| m.contains(a.as_str())))
            .collect()
    };
    let corpora: Vec<&str> = if args.len() < 2 {
        CORPORA.to_vec()
    } else {
        CORPORA
            .into_iter()
            .filter(|c| args.iter().any(|a| c.contains(a.as_str())))
            .collect()
    };

    for model in &models {
        let json = PathBuf::from(DATA_DIR).join(format!("{model}.json"));
        if !json.exists() {
            println!("-- {model}: {} missing, skipped", json.display());
            continue;
        }
        let tok = match Tokenizer::from_file(&json) {
            Ok(tok) => tok,
            Err(e) => {
                println!("-- {model}: could not load ({e}), skipped");
                continue;
            }
        };
        let tokens = vocab_of(&json);
        let store = Store::build(&tokens);
        let short = tokens.iter().filter(|(t, _)| t.len() <= 15).count();
        let (thin_b, fat_b, slim_b) = store.entries_bytes();
        println!("\n=== {model} ===");
        println!(
            "{} tokens, {:.1}% fit in 15 bytes | slab {:.2} MB | entries: thin {:.2} MB, fat {:.2} MB (+{:.2}), slim {:.2} MB (+{:.2})",
            tokens.len(),
            100.0 * short as f64 / tokens.len() as f64,
            store.slab.len() as f64 / 1e6,
            thin_b as f64 / 1e6,
            fat_b as f64 / 1e6,
            (fat_b - thin_b) as f64 / 1e6,
            slim_b as f64 / 1e6,
            (slim_b - thin_b) as f64 / 1e6,
        );

        for corpus in &corpora {
            let path = PathBuf::from(DATA_DIR)
                .join("fixtures/lang")
                .join(format!("{corpus}.txt"));
            let path = if path.exists() {
                path
            } else {
                PathBuf::from(DATA_DIR)
                    .join("fixtures/modalities")
                    .join(format!("{corpus}.txt"))
            };
            let Ok(text) = std::fs::read_to_string(&path) else {
                println!("  {corpus}: missing, skipped");
                continue;
            };
            let (buf, spans) = word_stream(&tok, &text);
            if spans.is_empty() {
                println!("  {corpus}: no words, skipped");
                continue;
            }

            let variants: [(&str, Lookup); 5] = [
                ("today   hash bytes  + slab memcmp    ", v_today),
                ("trick 1 hash packed + slab memcmp    ", v_packed_hash),
                ("1 + 2   fat, window from caller      ", v_fat),
                ("1 + 2   fat, no window (memcpy pack) ", v_fat_owned),
                ("1 + 2   slim, window from caller     ", v_slim),
            ];
            let mut best = [f64::MAX; 5];
            let mut sums = [0u64; 5];
            let mut hits = 0usize;
            for _ in 0..REPS {
                for (i, (_, f)) in variants.iter().enumerate() {
                    let mut sum = 0u64;
                    let mut found = 0usize;
                    for &(off, len) in &spans {
                        let q = &buf[off..off + len];
                        let head: Option<&[u8; 16]> = buf[off..].first_chunk();
                        if let Some(id) = black_box(f(&store, black_box(q), head)) {
                            sum += id as u64 + 1;
                            found += 1;
                        }
                    }
                    let t = Instant::now();
                    for &(off, len) in &spans {
                        let q = &buf[off..off + len];
                        let head: Option<&[u8; 16]> = buf[off..].first_chunk();
                        sum += black_box(f(&store, black_box(q), head)).map_or(0, |id| id as u64);
                    }
                    let ns = t.elapsed().as_secs_f64() * 1e9 / spans.len() as f64;
                    if ns < best[i] {
                        best[i] = ns;
                    }
                    sums[i] = sum;
                    hits = found;
                }
            }

            // The shipped store, as a fidelity control on the copies above.
            let shipped = BucketVocabStore::build(tokens.clone());
            let mut shipped_best = f64::MAX;
            let mut shipped_sum = 0u64;
            for _ in 0..REPS {
                let mut sum = 0u64;
                for &(off, len) in &spans {
                    sum += black_box(
                        shipped
                            .get_bytes(black_box(&buf[off..off + len]), buf[off..].first_chunk()),
                    )
                    .map_or(0, |id| id as u64 + 1);
                }
                let t = Instant::now();
                for &(off, len) in &spans {
                    sum += black_box(
                        shipped
                            .get_bytes(black_box(&buf[off..off + len]), buf[off..].first_chunk()),
                    )
                    .map_or(0, |id| id as u64);
                }
                let ns = t.elapsed().as_secs_f64() * 1e9 / spans.len() as f64;
                if ns < shipped_best {
                    shipped_best = ns;
                }
                shipped_sum = sum;
            }

            println!(
                "  {corpus}: {} words, {:.1}% in vocab",
                spans.len(),
                100.0 * hits as f64 / spans.len() as f64
            );
            for (i, (name, _)) in variants.iter().enumerate() {
                println!(
                    "      {name}  {:6.2} ns  {:+6.1}%",
                    best[i],
                    100.0 * (best[i] - best[0]) / best[0]
                );
            }
            println!(
                "      shipped BucketVocabStore (control)  {shipped_best:6.2} ns  {:+6.1}%",
                100.0 * (shipped_best - best[0]) / best[0]
            );
            for i in 1..5 {
                assert_eq!(sums[i], sums[0], "{} returned different ids", variants[i].0);
            }
            assert_eq!(
                shipped_sum, sums[0],
                "the copies disagree with the shipped store"
            );
        }
    }
}
