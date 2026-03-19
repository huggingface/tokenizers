use std::cell::Cell;
use std::error::Error;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

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

// Pattern for &SysRegex is implemented generically in tokenizer/pattern.rs
// using the find_iter() method defined above.
