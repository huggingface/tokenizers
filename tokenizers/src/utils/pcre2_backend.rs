use crate::tokenizer::pattern::Pattern;
use crate::utils::parallelism::{current_num_threads, get_parallelism, MaybeParallelIterator};
use crate::Offsets;
use std::cell::Cell;
use std::error::Error;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

/// Minimum chunk size (bytes) for parallel regex matching.
/// Parallel matching only triggers for inputs at least twice this size.
const MIN_CHUNK_SIZE: usize = 4 * 1024;

/// Overlap added to each chunk so matches that start near a boundary are still
/// seen in full by the owning chunk.
const CHUNK_OVERLAP: usize = 1024;

/// Cap the pool size — more copies waste memory without benefit since
/// concurrent encode calls on the same tokenizer are typically limited
/// by the number of physical cores doing real work.
const MAX_POOL_SIZE: usize = 32;

/// Number of pre-compiled regex copies.
fn pool_size() -> usize {
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::thread::available_parallelism()
            .map(|n| n.get().min(MAX_POOL_SIZE))
            .unwrap_or(1)
    })
}

/// Global counter for assigning thread-local pool indices.
static THREAD_COUNTER: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// Each thread gets a stable index into the regex pool.
    static THREAD_INDEX: Cell<usize> = Cell::new(
        THREAD_COUNTER.fetch_add(1, Ordering::Relaxed)
    );
}

#[inline]
fn thread_index(pool_len: usize) -> usize {
    THREAD_INDEX.with(|c| c.get()) % pool_len
}

/// A single PCRE2 regex instance. Each instance maintains its own DFA cache,
/// so sharing across threads causes cache thrashing.
struct Pcre2Regex {
    inner: pcre2::bytes::Regex,
}

impl Pcre2Regex {
    fn compile(pattern: &str) -> Result<Self, Box<dyn Error + Send + Sync + 'static>> {
        let inner = pcre2::bytes::RegexBuilder::new()
            .utf(true)
            .ucp(true)
            .jit_if_available(true)
            .build(pattern)?;
        Ok(Self { inner })
    }

    fn find_at(&self, text: &[u8], offset: usize) -> Option<(usize, usize)> {
        match self.inner.find_at(text, offset) {
            Ok(Some(m)) => Some((m.start(), m.end())),
            _ => None,
        }
    }
}

// Safety: pcre2::bytes::Regex is Send+Sync. Each Pcre2Regex instance has its
// own match context, so concurrent find_at calls on *different* instances are safe.
unsafe impl Send for Pcre2Regex {}
unsafe impl Sync for Pcre2Regex {}

/// PCRE2-backed regex with JIT compilation and per-thread copies.
///
/// Pre-compiles a pool of independent PCRE2 regex instances at construction time
/// (capped at 32). Each thread picks its own copy via a stable thread-local index,
/// avoiding DFA cache contention under concurrent use.
///
/// Falls back to `fancy_regex` at runtime if PCRE2 compilation fails for a
/// particular pattern.
pub struct SysRegex {
    /// Per-thread PCRE2 instances. None if PCRE2 compilation failed.
    pcre2_pool: Option<Vec<Pcre2Regex>>,
    /// Fallback regex, always available.
    fallback: fancy_regex::Regex,
}

impl std::fmt::Debug for SysRegex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.pcre2_pool.is_some() {
            write!(f, "SysRegex(pcre2-jit)")
        } else {
            write!(f, "SysRegex(fancy-regex-fallback)")
        }
    }
}

impl SysRegex {
    pub fn new(regex_str: &str) -> Result<Self, Box<dyn Error + Send + Sync + 'static>> {
        // Always compile fancy-regex as fallback
        let fallback = fancy_regex::Regex::new(regex_str)?;

        // Try PCRE2 — compile N independent copies for per-thread use
        let n = pool_size();
        let pcre2_pool = (0..n)
            .map(|_| Pcre2Regex::compile(regex_str))
            .collect::<Result<Vec<_>, _>>()
            .ok();

        Ok(Self {
            pcre2_pool,
            fallback,
        })
    }

    pub fn find_iter<'r, 't>(&'r self, inside: &'t str) -> Matches<'r, 't> {
        if let Some(pool) = &self.pcre2_pool {
            let idx = thread_index(pool.len());
            Matches::Pcre2(Pcre2Matches {
                regex: &pool[idx],
                text: inside.as_bytes(),
                offset: 0,
            })
        } else {
            Matches::Fancy(FancyMatches(self.fallback.find_iter(inside)))
        }
    }

    pub fn find_matches(
        &self,
        inside: &str,
    ) -> Result<Vec<(Offsets, bool)>, Box<dyn Error + Send + Sync + 'static>> {
        if inside.is_empty() {
            return Ok(vec![((0, 0), false)]);
        }

        let matches = match &self.pcre2_pool {
            Some(pool) if should_parallelize(inside.len(), pool.len()) => {
                find_matches_pcre2_parallel(inside, pool)?
            }
            Some(pool) => find_matches_pcre2(inside, 0, &pool[0])?,
            None => find_matches_fancy(inside, &self.fallback),
        };

        Ok(matches_to_splits(&matches, inside.len()))
    }
}

// ---------------------------------------------------------------------------
// Match iterators
// ---------------------------------------------------------------------------

pub enum Matches<'r, 't> {
    Pcre2(Pcre2Matches<'r, 't>),
    Fancy(FancyMatches<'r, 't>),
}

impl Iterator for Matches<'_, '_> {
    type Item = (usize, usize);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Matches::Pcre2(m) => m.next(),
            Matches::Fancy(m) => m.next(),
        }
    }
}

pub struct Pcre2Matches<'r, 't> {
    regex: &'r Pcre2Regex,
    text: &'t [u8],
    offset: usize,
}

impl Iterator for Pcre2Matches<'_, '_> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset > self.text.len() {
            return None;
        }
        let (start, end) = self.regex.find_at(self.text, self.offset)?;
        // Advance past this match (handle zero-length matches)
        if end == self.offset {
            self.offset += 1;
            // Skip to next valid UTF-8 boundary
            while self.offset < self.text.len() && (self.text[self.offset] & 0xC0) == 0x80 {
                self.offset += 1;
            }
        } else {
            self.offset = end;
        }
        Some((start, end))
    }
}

pub struct FancyMatches<'r, 't>(fancy_regex::Matches<'r, 't>);

impl Iterator for FancyMatches<'_, '_> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        match self.0.next() {
            Some(Ok(m)) => Some((m.start(), m.end())),
            _ => None,
        }
    }
}

impl Pattern for &pcre2::bytes::Regex {
    fn find_matches(
        &self,
        inside: &str,
    ) -> Result<Vec<(Offsets, bool)>, Box<dyn Error + Send + Sync + 'static>> {
        if inside.is_empty() {
            return Ok(vec![((0, 0), false)]);
        }

        let mut prev = 0;
        let mut splits = Vec::with_capacity(inside.len());
        let text = inside.as_bytes();
        let mut offset = 0;
        while offset <= text.len() {
            match self.find_at(text, offset) {
                Ok(Some(m)) => {
                    let start = m.start();
                    let end = m.end();
                    if prev != start {
                        splits.push(((prev, start), false));
                    }
                    splits.push(((start, end), true));
                    prev = end;
                    offset = advance_after_match(text, offset, end);
                }
                Ok(None) => break,
                Err(err) => return Err(Box::new(err)),
            }
        }
        if prev != inside.len() {
            splits.push(((prev, inside.len()), false));
        }
        Ok(splits)
    }
}

#[inline]
fn should_parallelize(input_len: usize, pool_len: usize) -> bool {
    get_parallelism() && pool_len >= 2 && input_len >= MIN_CHUNK_SIZE * 2
}

fn find_matches_fancy(inside: &str, regex: &fancy_regex::Regex) -> Vec<(usize, usize)> {
    let mut matches = Vec::new();
    for matched in regex.find_iter(inside) {
        match matched {
            Ok(m) if m.start() != m.end() => matches.push((m.start(), m.end())),
            Ok(_) => {}
            Err(_) => break,
        }
    }
    matches
}

fn find_matches_pcre2(
    text: &str,
    base: usize,
    regex: &Pcre2Regex,
) -> Result<Vec<(usize, usize)>, Box<dyn Error + Send + Sync + 'static>> {
    let bytes = text.as_bytes();
    let mut matches = Vec::new();
    let mut pos = 0;

    while pos <= bytes.len() {
        match regex.inner.find_at(bytes, pos) {
            Ok(Some(m)) => {
                if m.start() != m.end() {
                    matches.push((base + m.start(), base + m.end()));
                }
                pos = advance_after_match(bytes, pos, m.end());
            }
            Ok(None) => break,
            Err(err) => return Err(Box::new(err)),
        }
    }

    Ok(matches)
}

fn find_matches_pcre2_parallel(
    text: &str,
    pool: &[Pcre2Regex],
) -> Result<Vec<(usize, usize)>, Box<dyn Error + Send + Sync + 'static>> {
    let n_chunks = current_num_threads()
        .min(text.len() / MIN_CHUNK_SIZE)
        .min(pool.len())
        .max(2);
    let nominal = text.len() / n_chunks;

    let mut auth = vec![0usize];
    for i in 1..n_chunks {
        let boundary = snap_char_ceil(text, i * nominal);
        if boundary > *auth.last().unwrap() && boundary < text.len() {
            auth.push(boundary);
        }
    }
    auth.push(text.len());

    let actual = auth.len() - 1;
    if actual < 2 {
        return find_matches_pcre2(text, 0, &pool[0]);
    }

    let chunk_results = (0..actual)
        .into_maybe_par_iter_cond(actual >= 2)
        .map(|i| {
            let auth_start = auth[i];
            let auth_end = auth[i + 1];
            let chunk_end = snap_char_ceil(text, (auth_end + CHUNK_OVERLAP).min(text.len()));
            let chunk = &text[auth_start..chunk_end];
            let matches = find_matches_pcre2(chunk, auth_start, &pool[i])?;
            Ok::<_, Box<dyn Error + Send + Sync + 'static>>(
                matches
                    .into_iter()
                    .filter(|&(start, _)| start < auth_end)
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    merge_chunk_matches(chunk_results, |pos| {
        find_next_match_from(text, pos, &pool[0])
    })
}

fn find_next_match_from(
    text: &str,
    pos: usize,
    regex: &Pcre2Regex,
) -> Result<Option<(usize, usize)>, Box<dyn Error + Send + Sync + 'static>> {
    let bytes = text.as_bytes();
    let mut offset = pos;
    loop {
        if offset >= bytes.len() {
            return Ok(None);
        }
        match regex.inner.find_at(bytes, offset) {
            Ok(Some(m)) if m.start() != m.end() => return Ok(Some((m.start(), m.end()))),
            Ok(Some(m)) => {
                offset = advance_after_match(bytes, offset, m.end());
            }
            Ok(None) => return Ok(None),
            Err(err) => return Err(Box::new(err)),
        }
    }
}

fn merge_chunk_matches(
    chunks: Vec<Vec<(usize, usize)>>,
    mut find_from: impl FnMut(
        usize,
    )
        -> Result<Option<(usize, usize)>, Box<dyn Error + Send + Sync + 'static>>,
) -> Result<Vec<(usize, usize)>, Box<dyn Error + Send + Sync + 'static>> {
    let total = chunks.iter().map(Vec::len).sum();
    let mut flat = Vec::with_capacity(total);
    for chunk in chunks {
        flat.extend(chunk);
    }

    if flat.is_empty() {
        return Ok(flat);
    }

    let mut result = Vec::with_capacity(flat.len());
    let mut prev_end = 0;
    let mut idx = 0;

    while idx < flat.len() {
        if flat[idx].0 >= prev_end {
            result.push(flat[idx]);
            prev_end = flat[idx].1;
            idx += 1;
            continue;
        }

        let mut max_ghost_end = 0usize;
        while idx < flat.len() && flat[idx].0 < prev_end {
            max_ghost_end = max_ghost_end.max(flat[idx].1);
            idx += 1;
        }

        if max_ghost_end > prev_end {
            if let Some(&(trunc_start, _)) = result.last() {
                result.pop();
                if let Some((start, end)) = find_from(trunc_start)? {
                    result.push((start, end));
                    prev_end = end;
                } else {
                    prev_end = result.last().map_or(0, |&(_, end)| end);
                }
            }

            while idx < flat.len() && flat[idx].0 < prev_end {
                idx += 1;
            }
        }

        if idx < flat.len() && flat[idx].0 > prev_end {
            let remaining = &flat[idx..];
            let mut pos = prev_end;

            loop {
                match find_from(pos)? {
                    Some((start, end)) => {
                        let limit = remaining.len().min(64);
                        if let Some(offset) =
                            remaining[..limit].iter().position(|&m| m == (start, end))
                        {
                            idx += offset;
                            break;
                        }
                        result.push((start, end));
                        prev_end = end;
                        pos = end;
                    }
                    None => {
                        idx = flat.len();
                        break;
                    }
                }
            }
        }
    }

    Ok(result)
}

fn matches_to_splits(matches: &[(usize, usize)], input_len: usize) -> Vec<(Offsets, bool)> {
    let mut prev = 0;
    let mut splits = Vec::with_capacity(matches.len() * 2 + 1);
    for &(start, end) in matches {
        if prev != start {
            splits.push(((prev, start), false));
        }
        splits.push(((start, end), true));
        prev = end;
    }
    if prev != input_len {
        splits.push(((prev, input_len), false));
    }
    splits
}

#[inline]
fn advance_after_match(bytes: &[u8], current: usize, end: usize) -> usize {
    if end != current {
        return end;
    }

    let mut next = current + 1;
    while next < bytes.len() && (bytes[next] & 0xC0) == 0x80 {
        next += 1;
    }
    next
}

#[inline]
fn snap_char_ceil(text: &str, mut pos: usize) -> usize {
    while pos < text.len() && !text.is_char_boundary(pos) {
        pos += 1;
    }
    pos
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_input_parallel_matches_sequential() {
        let regex = SysRegex::new(
            r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
        )
        .unwrap();

        let input = format!(
            "{} {} {}{}",
            "a".repeat(MIN_CHUNK_SIZE + 321),
            "123".repeat(MIN_CHUNK_SIZE / 3),
            "!".repeat(MIN_CHUNK_SIZE + 137),
            " tail"
        );

        let sequential = match &regex.pcre2_pool {
            Some(pool) => matches_to_splits(
                &find_matches_pcre2(&input, 0, &pool[0]).unwrap(),
                input.len(),
            ),
            None => panic!("PCRE2 compilation failed, cannot run parallel match test"),
        };

        assert_eq!(regex.find_matches(&input).unwrap(), sequential);
    }
}
