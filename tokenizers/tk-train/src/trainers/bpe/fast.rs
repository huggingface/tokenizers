//! The merge loop on a fixed-coordinate corpus (the representation of Yikai Liao's PR #2501, without its
//! specializations).
//!
//! Every symbol of every word has one slot in a flat corpus, and slots never move. A live token stores its ID in
//! its first and last slot, and every token with a given ID covers the same number of slots (`span`), so from a
//! token's first slot the next token starts at `first + span`, and from a token's last slot the previous token ends
//! one slot before. Merging two tokens rewrites the two outer endpoints and marks the two inner ones `DEAD`; nothing
//! else in the word moves. Each pair keeps the slots where it was born (its occurrences); an occurrence is checked
//! against the corpus when used, so lists never need cleaning. Pair counts are exact; the queue holds upper bounds
//! and is repaired on pop, as `do_train`'s, so the merges are the same: highest count first, ties to the smaller
//! pair.
//!
//! A rule with many occurrences is merged in parallel: the corpus is cut at word boundaries into chunks, each task
//! merges the occurrences in its chunk and sorts its count changes by shard of the pair, and each shard of pairs
//! then takes in its changes from every chunk on its own task.
use super::Pair;
use ahash::AHashMap;
use compact_str::CompactString;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use tk_encode::parallelism::*;

const SEP: u32 = u32::MAX; // between words
const DEAD: u32 = u32::MAX - 1; // inner slot of a merged token
const SHARDS: usize = 64;
/// Batches with at least this many occurrences are merged in parallel.
const PARALLEL: usize = 4096;
/// At most this many rules per batch.
const BATCH: usize = 256;

type Queue = BinaryHeap<(i64, Reverse<u64>)>;

#[inline]
fn key(a: u32, b: u32) -> u64 {
    ((a as u64) << 32) | b as u64
}

#[inline]
fn shard(k: u64) -> usize {
    (k.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 58) as usize % SHARDS
}

/// One chunk's count changes: a local index of the pairs it touched (so each pair is looked up globally once), its
/// change per pair and the slots where pairs were born. `finish` lists them by shard of the pair: (pair, change,
/// births `at[from..to]`). Kept from batch to batch, so its allocations are reused.
#[derive(Default)]
struct Collect {
    index: AHashMap<u64, u32>,
    keys: Vec<u64>,
    sum: Vec<i64>,
    born: Vec<(u32, u32)>,
    start: Vec<u32>,
    at: Vec<u32>,
    by: Vec<Vec<(u64, i64, u32, u32)>>,
}

impl Collect {
    #[inline]
    fn add(&mut self, k: u64, d: i64, at: u32) {
        let n = self.keys.len() as u32;
        let i = *self.index.entry(k).or_insert(n);
        if i == n {
            self.keys.push(k);
            self.sum.push(0);
        }
        self.sum[i as usize] += d;
        if d > 0 {
            self.born.push((i, at));
        }
    }

    fn finish(&mut self) {
        // the births grouped by pair (counting sort)
        let n = self.keys.len();
        self.start.clear();
        self.start.resize(n + 1, 0);
        self.born
            .iter()
            .for_each(|&(i, _)| self.start[i as usize + 1] += 1);
        for i in 0..n {
            self.start[i + 1] += self.start[i];
        }
        self.at.resize(self.born.len(), 0);
        for &(i, p) in &self.born {
            let f = &mut self.start[i as usize];
            self.at[*f as usize] = p;
            *f += 1;
        }
        // (each start[i] now ends pair i's births: pair i's births are at[start[i - 1]..start[i]])
        self.by.resize(SHARDS, Vec::new());
        self.by.iter_mut().for_each(Vec::clear);
        for i in 0..n {
            let (k, d) = (self.keys[i], self.sum[i]);
            let from = if i == 0 { 0 } else { self.start[i - 1] };
            if d != 0 || from != self.start[i] {
                self.by[shard(k)].push((k, d, from, self.start[i]));
            }
        }
        self.index.clear();
        self.keys.clear();
        self.sum.clear();
        self.born.clear();
    }
}

/// The pairs of one shard: count and occurrences, by a dense index. `queued` remembers the last rule that queued
/// the pair, so a rule queues each pair once.
#[derive(Default)]
struct Shard {
    index: AHashMap<u64, u32>,
    keys: Vec<u64>,
    count: Vec<i64>,
    occ: Vec<Vec<u32>>,
    queued: Vec<u32>,
}

impl Shard {
    #[inline]
    fn get(&mut self, k: u64) -> usize {
        let n = self.count.len() as u32;
        let i = *self.index.entry(k).or_insert(n);
        if i == n {
            self.keys.push(k);
            self.count.push(0);
            self.occ.push(Vec::new());
            self.queued.push(u32::MAX);
        }
        i as usize
    }

    /// Take in the changes of shard `s` from every part (rule number `rule`); the pairs born are queued again (at
    /// their final count, once).
    fn take(
        &mut self,
        s: usize,
        parts: &[Collect],
        rule: u32,
        grown: &mut Vec<(i64, Reverse<u64>)>,
    ) {
        let mut born = Vec::new();
        for part in parts {
            for &(k, d, from, to) in &part.by[s] {
                let i = self.get(k);
                self.count[i] += d;
                if from != to {
                    self.occ[i].extend_from_slice(&part.at[from as usize..to as usize]);
                    if self.queued[i] != rule {
                        self.queued[i] = rule;
                        born.push(i);
                    }
                }
            }
        }
        grown.extend(
            born.into_iter()
                .map(|i| (self.count[i], Reverse(self.keys[i]))),
        );
    }
}

/// Words are sorted by descending weight, so the weight is a step function of the slot: `step` holds the first slot
/// with each weight (few distinct weights, a lookup stays in cache).
struct Weights {
    step: Vec<u32>,
    weight: Vec<i64>,
}

impl Weights {
    #[inline]
    fn at(&self, p: u32) -> i64 {
        self.weight[self.step.partition_point(|&s| s <= p) - 1]
    }
}

/// Trains on `words` (each with its weight): adds their alphabet and `initial_alphabet` to `vocab` / `names`
/// (which hold the special tokens), in code point order, then merges until `vocab` reaches `vocab_size` or no pair
/// has `min_frequency` occurrences. Returns the merges in order, with their result ID.
pub(super) fn train(
    words: Vec<(&str, u64)>,
    initial_alphabet: &ahash::AHashSet<char>,
    vocab: &mut AHashMap<CompactString, u32>,
    names: &mut Vec<CompactString>,
    vocab_size: usize,
    min_frequency: u64,
) -> Vec<(Pair, u32)> {
    let threads = current_num_threads().max(1);
    let group = words.len().div_ceil(threads * 4).max(1);
    // the alphabet
    let mut alphabet: Vec<char> = words
        .maybe_par_chunks(group)
        .map(|ws| {
            ws.iter()
                .flat_map(|(w, _)| w.chars())
                .collect::<ahash::AHashSet<char>>()
        })
        .reduce(ahash::AHashSet::new, |mut x, y| {
            x.extend(y);
            x
        })
        .into_iter()
        .chain(initial_alphabet.iter().copied())
        .collect();
    alphabet.sort_unstable();
    alphabet.dedup();
    for c in alphabet {
        let s = CompactString::from(&*c.encode_utf8(&mut [0; 4]));
        if !vocab.contains_key(&s) {
            vocab.insert(s.clone(), names.len() as u32);
            names.push(s);
        }
    }
    // char -> ID: a table for the first 64k code points, a map beyond
    let (mut low, mut high) = (vec![u32::MAX; 1 << 16], AHashMap::new());
    for (s, &id) in vocab.iter() {
        let mut cs = s.chars();
        if let (Some(c), None) = (cs.next(), cs.next()) {
            if (c as u32) < 1 << 16 {
                low[c as usize] = id;
            } else {
                high.insert(c, id);
            }
        }
    }
    let char_id = |c: char| {
        if (c as u32) < 1 << 16 {
            low[c as usize]
        } else {
            high[&c]
        }
    };
    let mut span = vec![1; names.len()];
    // the corpus: frequent words first, so the slots most merges visit stay close together; each group of words
    // filled on its own task
    let mut words: Vec<_> = words
        .into_iter()
        .filter(|(w, _)| w.chars().nth(1).is_some())
        .collect();
    words.sort_unstable_by_key(|(_, n)| Reverse(*n));
    let groups: Vec<&[(&str, u64)]> = words.chunks(group).collect();
    let sizes: Vec<usize> = groups
        .maybe_par_iter()
        .map(|g| g.iter().map(|(w, _)| w.chars().count() + 1).sum())
        .collect();
    let mut tok = vec![SEP; 1 + sizes.iter().sum::<usize>()];
    let mut rest = &mut tok[1..];
    let mut parts = Vec::with_capacity(groups.len());
    for (g, &n) in groups.iter().zip(&sizes) {
        let (head, tail) = std::mem::take(&mut rest).split_at_mut(n);
        parts.push((*g, head));
        rest = tail;
    }
    parts.into_maybe_par_iter().for_each(|(g, out)| {
        let mut j = 0;
        for (w, _) in g {
            for c in w.chars() {
                out[j] = char_id(c);
                j += 1;
            }
            j += 1; // SEP
        }
    });
    let mut ws = Weights {
        step: Vec::new(),
        weight: Vec::new(),
    };
    let mut p = 1;
    for (w, n) in &words {
        if ws.weight.last() != Some(&(*n as i64)) {
            ws.step.push(p as u32);
            ws.weight.push(*n as i64);
        }
        p += w.chars().count() + 1;
    }

    // chunks for parallel work: cut at word boundaries, one per thread
    let mut cuts = vec![0];
    for k in 1..threads {
        let mut x = (tok.len() * k / threads).max(cuts[cuts.len() - 1] + 1);
        while x < tok.len() && tok[x] != SEP {
            x += 1;
        }
        if x < tok.len() {
            cuts.push(x);
        }
    }
    cuts.push(tok.len());
    let mut pairs: Vec<Shard> = (0..SHARDS).map(|_| Shard::default()).collect();
    let mut queue = Queue::new();
    // initial pairs: each chunk's pairs as changes by shard, then each shard takes in all chunks
    let mut pool: Vec<Collect> = (0..cuts.len() - 1).map(|_| Collect::default()).collect();
    chunks(&mut tok, &cuts)
        .into_iter()
        .zip(pool.iter_mut())
        .collect::<Vec<_>>()
        .into_maybe_par_iter()
        .for_each(|((base, tok), c)| {
            for p in 1..tok.len().saturating_sub(1) {
                let (l, r) = (tok[p], tok[p + 1]);
                if l != SEP && r != SEP {
                    let at = (base + p) as u32;
                    c.add(key(l, r), ws.at(at), at);
                }
            }
            c.finish();
        });
    commit(&mut pairs, &pool, 0, &mut queue);
    let floor = min_frequency.max(1) as i64;
    let mut merges = Vec::new();
    let mut done = false;
    while !done && vocab.len() < vocab_size {
        // A batch: the next rules in queue order, as long as they can be merged together with the same result as
        // one after the other (Yikai Liao's ordered-prefix batching, PR #2501 DESIGN.md "Why batching preserves
        // rule priority"): new result IDs, no rule (A, A), no rule whose left token is an earlier rule's right
        // token or the other way round. Then no rule changes a later rule's count, and no pair born in the batch
        // can rank before a later rule.
        let mut batch: Vec<(u32, u32, u32, Vec<u32>)> = Vec::new();
        let mut fresh: AHashMap<CompactString, ()> = AHashMap::new();
        while batch.len() < BATCH && vocab.len() + batch.len() < vocab_size {
            let Some((n, Reverse(k))) = queue.pop() else {
                done = true;
                break;
            };
            let sh = &mut pairs[shard(k)];
            let i = sh.index[&k] as usize;
            if n != sh.count[i] {
                if sh.count[i] > 0 {
                    queue.push((sh.count[i], Reverse(k)));
                }
                continue;
            }
            if n < floor {
                queue.push((n, Reverse(k)));
                done = true;
                break;
            }
            let (a, b) = ((k >> 32) as u32, k as u32);
            let s = compact_str::format_compact!("{}{}", names[a as usize], names[b as usize]);
            let alone = a == b || vocab.contains_key(&s) || fresh.contains_key(&s);
            let crossed = batch.iter().any(|&(l, r, _, _)| a == r || b == l);
            if !batch.is_empty() && (alone || crossed) {
                queue.push((n, Reverse(k)));
                break;
            }
            let id = match vocab.get(&s) {
                Some(&id) => id,
                None => {
                    fresh.insert(s.clone(), ());
                    let id = names.len() as u32;
                    names.push(s.clone());
                    vocab.insert(s, id);
                    span.push(span[a as usize] + span[b as usize]);
                    id
                }
            };
            merges.push(((a, b), id));
            let mut occ = std::mem::take(&mut sh.occ[i]);
            if a == b {
                // overlapping matches: left to right within a word, as `Word::merge`
                occ.sort_unstable();
            }
            batch.push((a, b, id, occ));
            if alone {
                break;
            }
        }
        if batch.is_empty() {
            break;
        }
        let rule = merges.len() as u32;
        let total: usize = batch.iter().map(|r| r.3.len()).sum();
        let span = &span[..];
        let parallel = threads > 1 && total >= PARALLEL && batch[0].0 != batch[0].1;
        // a chunk merges the occurrences in it of each rule, rule after rule: as one after the other
        let task = |((base, tok), c): ((usize, &mut [u32]), &mut Collect)| {
            let mut mine = Vec::new();
            for &(a, b, id, ref occ) in &batch {
                let end = base + tok.len();
                mine.clear();
                mine.extend(
                    occ.iter()
                        .copied()
                        .filter(|&p| base <= p as usize && (p as usize) < end),
                );
                merge(tok, base, span, &mine, (a, b, id), &ws, |k, d, at| {
                    c.add(k, d, at)
                });
            }
            c.finish();
        };
        let parts = if parallel {
            chunks(&mut tok, &cuts)
                .into_iter()
                .zip(pool.iter_mut())
                .collect::<Vec<_>>()
                .into_maybe_par_iter()
                .for_each(task);
            &pool[..]
        } else {
            task(((0, &mut tok[..]), &mut pool[0]));
            &pool[..1]
        };
        commit(&mut pairs, parts, rule, &mut queue);
    }
    merges
}

/// The corpus cut at `cuts`, each chunk with the slot it starts at.
fn chunks<'a>(tok: &'a mut [u32], cuts: &[usize]) -> Vec<(usize, &'a mut [u32])> {
    let mut v = Vec::with_capacity(cuts.len());
    let mut rest = tok;
    for k in 0..cuts.len() - 1 {
        let (head, tail) = std::mem::take(&mut rest).split_at_mut(cuts[k + 1] - cuts[k]);
        v.push((cuts[k], head));
        rest = tail;
    }
    v
}

/// Each shard takes in its changes from every chunk, one shard per task.
fn commit(pairs: &mut [Shard], parts: &[Collect], rule: u32, queue: &mut Queue) {
    let grown: Vec<Vec<(i64, Reverse<u64>)>> = pairs
        .iter_mut()
        .enumerate()
        .collect::<Vec<_>>()
        .into_maybe_par_iter()
        .map(|(s, sh)| {
            let mut grown = Vec::new();
            sh.take(s, parts, rule, &mut grown);
            grown
        })
        .collect();
    queue.extend(grown.into_iter().flatten());
}

/// Merge the pair (a, b) into `id` at the slots `occ` where it is still there; `tok` holds the slots from `base`
/// on (a slot past its end reads as a separator). The count changes of the pair and its neighbors go to `out`.
#[inline]
fn merge(
    tok: &mut [u32],
    base: usize,
    span: &[u32],
    occ: &[u32],
    (a, b, id): (u32, u32, u32),
    ws: &Weights,
    mut out: impl FnMut(u64, i64, u32),
) {
    for (j, &p) in occ.iter().enumerate() {
        if let Some(&f) = occ.get(j + 16) {
            prefetch(&tok[f as usize - base]);
        }
        let p = p as usize - base;
        if tok[p] != a {
            continue;
        }
        let q = p + span[a as usize] as usize;
        if tok.get(q) != Some(&b) {
            continue;
        }
        let e = q + span[b as usize] as usize - 1;
        let w = ws.at((base + p) as u32);
        let at = |x: usize| (base + x) as u32;
        out(key(a, b), -w, at(p));
        let l = tok[p - 1];
        if l != SEP {
            let pl = p - span[l as usize] as usize;
            out(key(l, a), -w, at(pl));
            out(key(l, id), w, at(pl));
        }
        let r = tok.get(e + 1).copied().unwrap_or(SEP);
        if r != SEP {
            out(key(b, r), -w, at(q));
            out(key(id, r), w, at(p));
        }
        tok[q - 1] = DEAD;
        tok[q] = DEAD;
        tok[p] = id;
        tok[e] = id;
    }
}

/// Start loading the cache line of `x` (no effect on results).
#[inline(always)]
fn prefetch<T>(x: &T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch(x as *const T as *const i8, std::arch::x86_64::_MM_HINT_T0);
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = x;
}
