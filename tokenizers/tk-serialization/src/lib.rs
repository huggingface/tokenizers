//! # The `.tok` v1 container
//!
//! A tokenizer file with no parser. Sections are byte images of arrays, so reading one is a bounds
//! check and a pointer cast.
//!
//! **The point is binary size, not load time.** A `tokenizer.json` can only be read by linking
//! `serde_json`, and once that is reachable it drags the whole JSON stack into every binary that
//! can load a tokenizer — measured at 583 KB gzipped on this workspace, 2.6x the size of the same
//! encoder with no parser reachable. A `.tok` is read with bounds checks and `copy_from_slice`, so
//! an inference build links neither the parser nor anything only the parser reaches.
//!
//! It follows that this format deliberately stores *only what a `tokenizer.json` stores* — the
//! vocabulary, the merges, the added tokens, and which pre-tokenizer to run. The derived tables
//! (internal-id map, merge grid, codepoint fold, perfect hashes) are rebuilt at load exactly as
//! they are today. Baking them too would save tens of milliseconds once per process, which is not
//! a problem anybody has; it would also freeze `tk-encode`'s internal layout into a file format,
//! which is a problem everybody would then have.
//!
//! This crate is the container and the schema: header, section table, aligned reader, section
//! kinds. It knows nothing about tokenizers. `tk-encode` reads and writes its own types against
//! these primitives, and `tk-convert` drives the write side from a legacy `tokenizer.json`.
//!
//! ## File layout
//!
//! ```text
//! 0                                                            file_len
//! ├─ Header (16 B) ─┬─ Section[n_sections] (16 B each) ─┬─ pad ─┬─ section data ─┤
//!                                                               ^ every section 64 B aligned
//! ```
//!
//! Little-endian only; a big-endian host is rejected at load. Offsets are `u32` — a tokenizer over
//! 4 GiB is not a thing we are going to support.
//!
//! ### Header — 16 bytes at offset 0
//!
//! | field        | type      | value                                       |
//! |--------------|-----------|---------------------------------------------|
//! | `magic`      | `[u8; 4]` | `b"TOK\x01"`                                |
//! | `n_sections` | `u16`     | number of section descriptors               |
//! | `version`    | `u16`     | [`VERSION`]                                 |
//! | `file_len`   | `u32`     | total file size in bytes                    |
//! | `_reserved`  | `u32`     | 0                                           |
//!
//! ### Section descriptor — 16 bytes, `n_sections` of them, right after the header
//!
//! | field    | type  | value                                     |
//! |----------|-------|-------------------------------------------|
//! | `kind`   | `u32` | one of [`kind`]                           |
//! | `offset` | `u32` | byte offset from file start, 64 B aligned |
//! | `len`    | `u32` | byte length of the section                |
//! | `_pad`   | `u32` | 0                                         |
//!
//! Descriptors are sorted by `kind`. Unknown kinds are skipped, which is the only forward
//! compatibility v1 offers; anything else is a new magic.

use core::mem::{align_of, size_of};

#[cfg(feature = "write")]
mod write;
#[cfg(feature = "write")]
pub use write::Writer;

/// `b"TOK\x01"` — the first four bytes of every `.tok` file.
pub const MAGIC: [u8; 4] = *b"TOK\x01";

/// Sections start on a multiple of this so the reader can reinterpret one as a slice of its
/// element type in place. 64 = a cache line on every target we care about.
pub const SECTION_ALIGN: usize = 64;

/// Format version. Nothing derived is stored, so this only moves when the section schema itself
/// changes — and v1 has no forward compatibility beyond skipping unknown section kinds, so a real
/// change is a new magic rather than a bump.
pub const VERSION: u16 = 1;

/// Section kinds. Reader and writer share these; each is the byte image of one array.
pub mod kind {
    /// One [`crate::Config`].
    pub const CONFIG: u32 = 1;
    /// `u8` — every vocabulary token's bytes, concatenated. Stored exactly as the model declares
    /// them, byte-level alphabet included, so the reader hands `tk-encode` what it expects.
    pub const VOCAB_SLAB: u32 = 2;
    /// [`crate::Entry`] — one per vocabulary token.
    pub const VOCAB_ENTRY: u32 = 3;
    /// `u32` pairs — `(left id, right id)` in rank order, so a merge's rank is its index.
    pub const MERGE_PAIRS: u32 = 4;
    /// `u8` — added and special token bytes.
    pub const ADDED_SLAB: u32 = 5;
    /// [`crate::AddedEntry`].
    pub const ADDED_ENTRY: u32 = 6;
    /// `u32` — ids the post-processor puts before the sequence.
    pub const POST_PREFIX: u32 = 7;
    /// `u32` — ids the post-processor puts after it.
    pub const POST_SUFFIX: u32 = 8;
    /// `u8` — the model's three optional strings, in [`crate::strings`] form and in order:
    /// `unk_token`, `continuing_subword_prefix`, `end_of_word_suffix`.
    pub const MODEL_STRINGS: u32 = 9;
    /// `u8` — the normalizer, in [`crate::strings`] form: `[kind, ...arguments]`. v1 knows one
    /// kind, `"replace"`, whose arguments are the literal pattern and its replacement. That is
    /// enough for the SentencePiece-style ` ` -> `U+2581` rewrite the gemma family ships, and it
    /// needs no regex engine.
    pub const NORMALIZER: u32 = 10;
    /// `u8` — [`crate::strings`] form, one entry: the literal pattern of a
    /// [`crate::pretok::LITERAL`] split. Its behaviour is in [`crate::Config::pretok_param`].
    pub const PRETOK_STRINGS: u32 = 11;
}

/// Which pre-tokenizer FSM to run. Stored in [`Config::pretok`].
pub mod pretok {
    /// No split: the whole segment is one pre-token.
    pub const NONE: u32 = 0;
    /// The GPT-2 / ByteLevel regex.
    pub const BYTE_LEVEL: u32 = 1;
    /// cl100k_base, i.e. Llama-3. [`crate::Config::pretok_param`] carries the digit cap.
    pub const CL100K: u32 = 2;
    /// o200k_base.
    pub const O200K: u32 = 3;
    /// Mistral tekken.
    pub const TEKKEN: u32 = 4;
    /// DeepSeek-V3/R1.
    pub const DEEPSEEK: u32 = 5;
    /// Split on a literal string, which needs no regex engine. The pattern is in
    /// [`crate::kind::PRETOK_STRINGS`] and the behaviour in [`crate::Config::pretok_param`], as a
    /// [`crate::behavior`] value.
    pub const LITERAL: u32 = 6;
}

/// How a [`pretok::LITERAL`] split treats its delimiter. Mirrors `SplitDelimiterBehavior`.
pub mod behavior {
    pub const REMOVED: u32 = 0;
    pub const ISOLATED: u32 = 1;
    pub const MERGED_WITH_PREVIOUS: u32 = 2;
    pub const MERGED_WITH_NEXT: u32 = 3;
    pub const CONTIGUOUS: u32 = 4;
}

/// [`Config::flags`] bits.
pub mod flag {
    /// A pre-token that is itself in the vocabulary skips the merge loop.
    pub const IGNORE_MERGES: u32 = 1 << 0;
    /// Special tokens in the input are encoded as ordinary text rather than carved out.
    pub const ENCODE_SPECIAL_TOKENS: u32 = 1 << 1;
    /// The pre-tokenizer ends in a `ByteLevel`, so the model seeds on bytes.
    pub const BYTE_LEVEL: u32 = 1 << 2;
    /// An out-of-vocabulary character falls back to its `<0xNN>` byte tokens.
    pub const BYTE_FALLBACK: u32 = 1 << 3;
    /// Consecutive unknown tokens collapse into one.
    pub const FUSE_UNK: u32 = 1 << 4;
    /// A [`crate::pretok::LITERAL`] split matches the gaps between its pattern rather than it.
    pub const PRETOK_INVERT: u32 = 1 << 5;
}

/// Length-prefixed string lists, the format's only variable-length text.
///
/// A handful of short strings — an unknown token, a normalizer's replacement — do not deserve a
/// section each, and they are read once at load, so there is nothing to gain from making them
/// castable. Each is a little-endian `u32` length followed by that many UTF-8 bytes.
pub mod strings {
    /// Append `value` to a string-list section body.
    pub fn push(out: &mut Vec<u8>, value: &str) {
        out.extend_from_slice(&(value.len() as u32).to_le_bytes());
        out.extend_from_slice(value.as_bytes());
    }

    /// Decode a string-list section body. Returns `None` if it is truncated or not UTF-8.
    pub fn parse(raw: &[u8]) -> Option<Vec<&str>> {
        let mut out = Vec::new();
        let mut at = 0usize;
        while at < raw.len() {
            let len = u32::from_le_bytes(raw.get(at..at + 4)?.try_into().ok()?) as usize;
            at += 4;
            out.push(core::str::from_utf8(raw.get(at..at + len)?).ok()?);
            at += len;
        }
        Some(out)
    }
}

/// [`AddedEntry::flags`] bits.
pub mod added_flag {
    pub const LSTRIP: u32 = 1 << 0;
    pub const RSTRIP: u32 = 1 << 1;
    pub const SPECIAL: u32 = 1 << 2;
    pub const SINGLE_WORD: u32 = 1 << 3;
    pub const NORMALIZED: u32 = 1 << 4;
}

/// The 16-byte file header.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Header {
    pub magic: [u8; 4],
    pub n_sections: u16,
    pub version: u16,
    pub file_len: u32,
    pub _reserved: u32,
}

/// One 16-byte section descriptor.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Section {
    pub kind: u32,
    pub offset: u32,
    pub len: u32,
    pub _pad: u32,
}

/// Everything about the tokenizer that is not an array: 32 bytes, no strings.
///
/// Note what is absent — no normalizer, no decoder, no truncation or padding policy. Byte-level
/// BPE has no normalizer, and the other two are caller policy rather than tokenizer identity.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Config {
    /// One of [`pretok`].
    pub pretok: u32,
    /// Pre-tokenizer parameter. Only [`pretok::CL100K`] uses it: rule 3's `\p{N}{1,cap}` bound
    /// (3 = cl100k/Llama-3, 1 = Qwen2, `u32::MAX` = unbounded). 0 elsewhere.
    pub pretok_param: u32,
    /// [`flag`] bits.
    pub flags: u32,
    /// Explicit, so `Config` has no implicit padding and its byte image is fully initialised.
    pub _pad0: u32,
    /// Bitmap of first bytes that can start an added token: bit `b` of `added_first[b / 64]`.
    /// One load per input byte rules out the added-token scan on ordinary text.
    pub added_first: [u64; 4],
}

/// One vocabulary token: a range into `VOCAB_SLAB` and the id it maps to.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Entry {
    pub start: u32,
    pub len: u32,
    pub id: u32,
}

/// One added or special token.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct AddedEntry {
    pub start: u32,
    pub len: u32,
    pub id: u32,
    /// [`added_flag`] bits.
    pub flags: u32,
}

// ── Reading ────────────────────────────────────────────────────────────────────────────────────

/// Everything that can go wrong opening a `.tok`. No `thiserror`, no `std::error::Error` chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Not a `.tok` file, or a version this build does not know.
    BadMagic,
    /// The file was written by a different version of the schema.
    Version { file: u16, expected: u16 },
    /// Truncated, overlapping or misaligned section table / section.
    Corrupt(&'static str),
    /// A section the reader requires is not in the file.
    MissingSection(u32),
    /// The buffer handed to [`Reader::new`] is not 8-byte aligned.
    Unaligned,
    /// Host is big-endian.
    BigEndian,
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::BadMagic => write!(f, "not a .tok v1 file"),
            Self::Version { file, expected } => {
                write!(f, ".tok is schema v{file}, this build reads v{expected}")
            }
            Self::Corrupt(what) => write!(f, "corrupt .tok: {what}"),
            Self::MissingSection(k) => write!(f, "corrupt .tok: missing section kind {k}"),
            Self::Unaligned => write!(f, ".tok buffer must be 8-byte aligned"),
            Self::BigEndian => write!(f, ".tok is little-endian only"),
        }
    }
}

impl std::error::Error for Error {}

/// A `.tok` read into memory, 8-byte aligned so sections can be reinterpreted in place.
///
/// Backed by a `Box<[u64]>` because that is the alignment the format needs and `Vec<u8>` does not
/// give it. An `mmap` is page-aligned and works with [`Reader::new`] directly.
pub struct TokFile {
    words: Box<[u64]>,
    len: usize,
}

impl TokFile {
    /// Read a `.tok` off disk into an aligned buffer.
    pub fn open(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        use std::io::Read;
        let mut file = std::fs::File::open(path)?;
        let len = file.metadata()?.len() as usize;
        let mut words = vec![0u64; len.div_ceil(8)].into_boxed_slice();
        // SAFETY: `words` owns `len.div_ceil(8) * 8 >= len` initialised bytes and `u64` has no
        // invalid bit patterns, so viewing it as `&mut [u8]` to fill is sound.
        let bytes =
            unsafe { core::slice::from_raw_parts_mut(words.as_mut_ptr().cast::<u8>(), len) };
        file.read_exact(bytes)?;
        Ok(Self { words, len })
    }

    /// Wrap bytes that are already 8-byte aligned (an `mmap`, or another `.tok` image).
    pub fn from_words(words: Box<[u64]>, len: usize) -> Self {
        Self { words, len }
    }

    /// The file bytes, 8-byte aligned.
    pub fn bytes(&self) -> &[u8] {
        // SAFETY: same provenance as the write above; `len` bytes are initialised.
        unsafe { core::slice::from_raw_parts(self.words.as_ptr().cast::<u8>(), self.len) }
    }

    /// Parse the section table. Borrows `self`, so no view can outlive the bytes.
    pub fn reader(&self) -> Result<Reader<'_>, Error> {
        Reader::new(self.bytes())
    }
}

/// A parsed section table over a `.tok` image. Handing out a section is a bounds and alignment
/// check, then a pointer cast — nothing is copied and nothing is allocated.
#[derive(Clone, Debug)]
pub struct Reader<'a> {
    raw: &'a [u8],
    table: &'a [Section],
    pub config: &'a Config,
}

impl<'a> Reader<'a> {
    /// Parse a `.tok` image. `raw` must be 8-byte aligned — use [`TokFile`] or an `mmap`.
    pub fn new(raw: &'a [u8]) -> Result<Self, Error> {
        if cfg!(target_endian = "big") {
            return Err(Error::BigEndian);
        }
        if raw.as_ptr() as usize % 8 != 0 {
            return Err(Error::Unaligned);
        }
        if raw.len() < size_of::<Header>() || raw[..4] != MAGIC {
            return Err(Error::BadMagic);
        }
        let header = cast::<Header>(raw, 0, size_of::<Header>())?[0];
        if header.version != VERSION {
            return Err(Error::Version {
                file: header.version,
                expected: VERSION,
            });
        }
        if header.file_len as usize > raw.len() {
            return Err(Error::Corrupt("file_len exceeds buffer"));
        }
        let table = cast::<Section>(
            raw,
            size_of::<Header>(),
            header.n_sections as usize * size_of::<Section>(),
        )?;

        let mut reader = Self {
            raw,
            table,
            // Placeholder: replaced immediately below, and `new` is the only way to build a
            // `Reader`, so no caller can observe it.
            config: &Config {
                pretok: 0,
                pretok_param: 0,
                flags: 0,
                _pad0: 0,
                added_first: [0; 4],
            },
        };
        let config = reader.require::<Config>(kind::CONFIG)?;
        if config.len() != 1 {
            return Err(Error::Corrupt("CONFIG must hold exactly one Config"));
        }
        reader.config = &config[0];
        Ok(reader)
    }

    /// A section as `&[T]`, or an empty slice if the file does not carry it.
    pub fn section<T: Copy>(&self, kind: u32) -> Result<&'a [T], Error> {
        match self.table.iter().find(|s| s.kind == kind) {
            Some(s) => cast(self.raw, s.offset as usize, s.len as usize),
            None => Ok(&[]),
        }
    }

    /// A section as `&[T]`, erroring if it is absent.
    pub fn require<T: Copy>(&self, kind: u32) -> Result<&'a [T], Error> {
        let s = self
            .table
            .iter()
            .find(|s| s.kind == kind)
            .ok_or(Error::MissingSection(kind))?;
        cast(self.raw, s.offset as usize, s.len as usize)
    }

    /// A section as a fixed-size array reference, erroring unless the length matches exactly.
    pub fn require_array<T: Copy, const N: usize>(&self, kind: u32) -> Result<&'a [T; N], Error> {
        let s = self.require::<T>(kind)?;
        s.try_into()
            .map_err(|_| Error::Corrupt("fixed-size section has the wrong length"))
    }
}

/// Reinterpret `raw[off .. off + len]` as `&[T]`, checking alignment, bounds and element fit.
fn cast<T: Copy>(raw: &[u8], off: usize, len: usize) -> Result<&[T], Error> {
    let end = off.checked_add(len).ok_or(Error::Corrupt("offset overflow"))?;
    if end > raw.len() {
        return Err(Error::Corrupt("section past end of file"));
    }
    if len % size_of::<T>() != 0 {
        return Err(Error::Corrupt("section length is not a multiple of its element size"));
    }
    let ptr = raw[off..].as_ptr();
    if (ptr as usize) % align_of::<T>() != 0 {
        return Err(Error::Corrupt("section misaligned"));
    }
    // SAFETY: bounds, alignment and element-size divisibility are all checked above. Every section
    // element is a `#[repr(C)]` aggregate of plain integers, so every bit pattern is valid `T`.
    Ok(unsafe { core::slice::from_raw_parts(ptr.cast::<T>(), len / size_of::<T>()) })
}

#[cfg(all(test, feature = "write"))]
mod tests {
    use super::*;

    /// Round-trips the container itself: three sections of different element types and alignments
    /// come back identical, an absent section reads empty, and a bad magic is refused.
    #[test]
    fn container_roundtrip() {
        let words: Vec<u64> = (0..37).map(|i| i * 0x0101_0101_0101_0101).collect();
        let halves: Vec<u16> = (0..999u16).collect();
        let config = Config {
            pretok: pretok::CL100K,
            pretok_param: 3,
            flags: flag::IGNORE_MERGES,
            _pad0: 0,
            added_first: [1, 2, 3, 4],
        };

        let mut w = Writer::new();
        w.push_one(kind::CONFIG, &config);
        w.push(kind::MERGE_PAIRS, &words);
        w.push(kind::ADDED_ENTRY, &halves);
        let image = w.finish();

        // Go through the aligned buffer a real load uses: a bare `Vec<u8>` is only 1-aligned.
        let file = TokFile::from_words(to_words(&image), image.len());
        let r = file.reader().unwrap();

        assert_eq!(r.config.pretok, pretok::CL100K);
        assert_eq!(r.config.pretok_param, 3);
        assert_eq!(r.config.added_first, [1, 2, 3, 4]);
        assert_eq!(r.require::<u64>(kind::MERGE_PAIRS).unwrap(), &words[..]);
        assert_eq!(r.require::<u16>(kind::ADDED_ENTRY).unwrap(), &halves[..]);
        assert_eq!(r.section::<u32>(kind::POST_PREFIX).unwrap(), &[] as &[u32]);
        assert_eq!(
            r.require::<u32>(kind::POST_PREFIX).unwrap_err(),
            Error::MissingSection(kind::POST_PREFIX)
        );

        let mut broken = image.clone();
        broken[1] = b'X';
        assert_eq!(
            Reader::new(to_words_ref(&broken)).unwrap_err(),
            Error::BadMagic
        );
    }

    fn to_words(bytes: &[u8]) -> Box<[u64]> {
        let mut w = vec![0u64; bytes.len().div_ceil(8)].into_boxed_slice();
        // SAFETY: `w` owns at least `bytes.len()` bytes of initialised `u64` storage.
        unsafe {
            core::slice::from_raw_parts_mut(w.as_mut_ptr().cast::<u8>(), bytes.len())
                .copy_from_slice(bytes);
        }
        w
    }

    /// Leaks a small aligned copy so the test can hold a `&[u8]` with no owner in scope.
    fn to_words_ref(bytes: &[u8]) -> &'static [u8] {
        let w = Box::leak(to_words(bytes));
        // SAFETY: `w` is 8-aligned and at least `bytes.len()` bytes long, and leaked, so 'static.
        unsafe { core::slice::from_raw_parts(w.as_ptr().cast::<u8>(), bytes.len()) }
    }
}
