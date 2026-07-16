//! Scan-based per-char normalizers — Lowercase, StripAccents, Nmt, and the fused Bert normalizer —
//! on the same layered-skip architecture as the forms, over baked per-rule property sets.
//!
//! The shape shared by all four: **find the next char the rule set touches** (ASCII rides an
//! in-register transform lane — the PR #2036 `|0x20` port — and non-ASCII a runtime 64-byte lead
//! mask + per-cp bitmap), copy everything before it verbatim, fix that one char up scalar, repeat.
//! Until the first real change the scan is a pure check and the input is returned `Cow::Borrowed`.
//!
//! Property sets are generated from the SAME predicates the legacy tk-encode normalizers run
//! (std `to_lowercase`, `unicode_categories`, `unicode_normalization_alignments`) — bug-compatible
//! by construction. A normalizer's set is an OR of rule bitmaps done once (`OnceLock`), its lead
//! mask derived from the result; astral membership rides one shared `(start, props)` RLE.
use crate::norm::{decode_cp, raw_extend};
use crate::tables::*;
use std::borrow::Cow;
use std::sync::OnceLock;

// astral property bits (`SCAN_ASTRAL` RLE values; keep in sync with `gen.rs` `props`)
const P_UPPER: u8 = 1;
const P_MN: u8 = 2;
const P_M: u8 = 4;
const P_CLEAN: u8 = 8;
const P_CJK: u8 = 16;
const P_STRIP: u8 = 32;
const P_WS: u8 = 64;

/// Astral props (RLE binary search — astral suspects are rare by construction).
#[inline]
fn astral_props(cp: u32) -> u8 {
    match SCAN_ASTRAL.binary_search_by(|&(s, _)| s.cmp(&cp)) {
        Ok(k) => SCAN_ASTRAL[k].1,
        Err(k) => SCAN_ASTRAL[k - 1].1,
    }
}

/// Is `cp` in the property? BMP: bitmap bit; astral: RLE props bit.
#[inline]
fn has(bmp: &[u64; 1024], astral_bit: u8, cp: u32) -> bool {
    if cp < 0x10000 {
        bmp[(cp >> 6) as usize] >> (cp & 63) & 1 != 0
    } else {
        astral_props(cp) & astral_bit != 0
    }
}

#[inline]
fn width(cp: u32) -> usize {
    if cp < 0x80 {
        1
    } else if cp < 0x800 {
        2
    } else if cp < 0x10000 {
        3
    } else {
        4
    }
}

/// A materialized scan set: the union bitmap of every rule the normalizer enables, its derived
/// lead mask (0/FF per lead byte, `vqtbl` fodder), and the astral policy — either a mask into the
/// baked `SCAN_ASTRAL` props, or `astral_all` (every astral char is a hit; for runtime sets).
pub(crate) struct Scan {
    set: [u64; 1024],
    lead: [u8; 64],
    astral: u8,
    astral_all: bool,
}

impl Scan {
    pub(crate) fn build_runtime(bmp: &[u64; 1024], astral_all: bool) -> Scan {
        let mut sc = Scan::build(&[bmp], 0);
        if astral_all {
            sc.astral_all = true;
            for l in 0x30..0x35 {
                sc.lead[l] = 0xFF;
            }
        }
        sc
    }
    fn build(sets: &[&[u64; 1024]], astral: u8) -> Scan {
        let mut set = [0u64; 1024];
        for s in sets {
            for (d, x) in set.iter_mut().zip(s.iter()) {
                *d |= x;
            }
        }
        let mut lead = [0u8; 64];
        // 2-byte lead L covers exactly bitmap word L & 0x1F; 3-byte lead L covers 64 words
        for l in 0xC2u8..=0xDF {
            if set[(l & 0x1F) as usize] != 0 {
                lead[(l - 0xC0) as usize] = 0xFF;
            }
        }
        for l in 0xE0u8..=0xEF {
            let base = ((l & 0x0F) as usize) << 6;
            if set[base..base + 64].iter().any(|&w| w != 0) {
                lead[(l - 0xC0) as usize] = 0xFF;
            }
        }
        if astral != 0 {
            // 4-byte lead = 0xF0 | cp >> 18: mark each lead whose cp range holds a relevant run
            for (k, &(s, p)) in SCAN_ASTRAL.iter().enumerate() {
                if p & astral == 0 {
                    continue;
                }
                let e = SCAN_ASTRAL.get(k + 1).map_or(0x110000, |&(n, _)| n);
                for l in (s >> 18)..=((e - 1) >> 18) {
                    lead[(0x30 + l) as usize] = 0xFF;
                }
            }
        }
        Scan {
            set,
            lead,
            astral,
            astral_all: false,
        }
    }
    #[inline]
    fn hit_bmp(&self, cp: u32) -> bool {
        self.set[(cp >> 6) as usize] >> (cp & 63) & 1 != 0
    }
    /// Set membership of any codepoint under this scan's astral policy.
    #[inline]
    pub(crate) fn contains(&self, cp: u32) -> bool {
        if cp < 0x10000 {
            self.hit_bmp(cp)
        } else {
            self.astral_all || astral_props(cp) & self.astral != 0
        }
    }
    /// Position of the next set char at/after byte `i` (always a char boundary), or `len`.
    /// Unlike the built-in normalizers (whose ASCII behavior is the policy lane), runtime sets may
    /// contain ASCII members, so this probes them too (`ASCII_SET`).
    pub(crate) fn next_member<const SIMD: bool>(&self, bytes: &[u8], i: usize) -> usize {
        let mut dummy = String::new();
        next_hit::<false, false, 0, SIMD, true>(bytes, i, &mut dummy, self).0
    }
}

/// Advance to the next set char (returned decoded) or ASCII policy stop (returned as the byte).
/// `LOWER`/`CLEAN` define the ASCII lane: `LOWER` flips `A..=Z`, `CLEAN` (1 = bert, 2 = nmt) folds
/// its whitespace to `' '` and STOPS at its removal bytes. In check mode (`!STORE`) any ASCII lane
/// change is itself a stop — the caller's borrow gate; in write mode it is transformed in place.
fn next_hit<
    const STORE: bool,
    const LOWER: bool,
    const CLEAN: u8,
    const SIMD: bool,
    const ASCII_SET: bool,
>(
    bytes: &[u8],
    mut i: usize,
    out: &mut String,
    sc: &Scan,
) -> (usize, u32) {
    let n = bytes.len();
    // scalar-first with streak escalation: dense hits (bert CJK padding, mark-heavy scripts) never
    // pay a kernel entry; a run of clean bytes escalates to the SIMD lane, and a suspect-lead probe
    // miss (E2-punctuation inside CJK, cased 2-byte scripts) drops back to scalar
    let mut streak = 0u32;
    while i < n {
        #[cfg(target_arch = "aarch64")]
        if SIMD && streak >= 8 {
            i = crate::simd_norm::scan_prefix::<STORE, LOWER, CLEAN, ASCII_SET>(
                bytes, i, out, &sc.lead, &sc.set,
            );
            streak = 0;
            if i >= n {
                break;
            }
        }
        let b = bytes[i];
        if b < 0x80 {
            if ASCII_SET && sc.hit_bmp(b as u32) {
                return (i, b as u32);
            }
            let fold = match CLEAN {
                1 => matches!(b, 9 | 10 | 13),
                2 => matches!(b, 9 | 10 | 12 | 13),
                _ => false,
            };
            let rm = match CLEAN {
                1 => (b < 0x20 && !fold) || b == 0x7F,
                2 => (b < 0x20 && !fold && b != 0) || b == 0x7F,
                _ => false,
            };
            if rm {
                return (i, b as u32);
            }
            let up = LOWER && b.is_ascii_uppercase();
            if !STORE && (up || fold) {
                return (i, b as u32);
            }
            if STORE {
                let t = if up {
                    b | 0x20
                } else if fold {
                    b' '
                } else {
                    b
                };
                // SAFETY: single ASCII byte keeps the String valid UTF-8; capacity reserved.
                unsafe { out.as_mut_vec().push(t) };
            }
            i += 1;
            streak += 1;
            continue;
        }
        if b < 0xC0 {
            // continuation byte: a clean-lead char split by the SIMD 16-byte stride — copy through
            if STORE {
                // SAFETY: verbatim byte of an already-verified char.
                unsafe { out.as_mut_vec().push(b) };
            }
            i += 1;
            streak += 1;
            continue;
        }
        let m = sc.lead[(b - 0xC0) as usize];
        let w = if b < 0xE0 {
            2
        } else if b < 0xF0 {
            3
        } else {
            4
        };
        if m != 0 {
            let (cp, _) = decode_cp(bytes, i);
            let hit = sc.contains(cp);
            if hit {
                return (i, cp);
            }
            streak = 0; // suspect neighbourhood: stay scalar
        } else {
            streak += w as u32;
        }
        if STORE {
            // SAFETY: verbatim whole char; capacity reserved by the driver invariant.
            unsafe { raw_extend(out, bytes.as_ptr().add(i), w, w) };
        }
        i += w;
    }
    (n, 0)
}

/// One scan-based normalizer: its set + what to do at a hit.
trait Rule {
    fn scan(&self) -> &'static Scan;
    /// Handle the hit char at `bytes[i..]` (`cp` pre-decoded; ASCII stops arrive as the byte),
    /// push its replacement (possibly nothing) onto `out`, and return the next input position.
    fn fixup(&self, bytes: &[u8], i: usize, cp: u32, out: &mut String) -> usize;
}

/// The shared driver: check-scan to the first change (borrow if none), then copy the verified
/// prefix and alternate write-through scans with per-hit fixups.
fn run<'a, const LOWER: bool, const CLEAN: u8, const SIMD: bool, R: Rule>(
    r: &R,
    input: &'a str,
) -> Cow<'a, str> {
    let bytes = input.as_bytes();
    let n = bytes.len();
    let sc = r.scan();
    let mut dummy = String::new();
    let (mut i, mut cp) = next_hit::<false, LOWER, CLEAN, SIMD, false>(bytes, 0, &mut dummy, sc);
    if i >= n {
        return Cow::Borrowed(input);
    }
    let mut out = String::with_capacity(n + n / 4 + 64);
    out.push_str(&input[..i]);
    loop {
        i = r.fixup(bytes, i, cp, &mut out);
        if i >= n {
            break;
        }
        // keep the raw-store slack invariant: capacity ≥ len + remaining + 32
        out.reserve(n - i + 32);
        let (pos, c2) = next_hit::<true, LOWER, CLEAN, SIMD, false>(bytes, i, &mut out, sc);
        if pos >= n {
            break;
        }
        (i, cp) = (pos, c2);
    }
    Cow::Owned(out)
}

// ── Lowercase ─────────────────────────────────────────────────────────────────────────────────────

struct Lower;
impl Rule for Lower {
    fn scan(&self) -> &'static Scan {
        static S: OnceLock<Scan> = OnceLock::new();
        S.get_or_init(|| Scan::build(&[&SCAN_UPPER], P_UPPER))
    }
    fn fixup(&self, _bytes: &[u8], i: usize, cp: u32, out: &mut String) -> usize {
        // every hit has a lowercase mapping ≠ itself (1..=3 chars)
        for l in char::from_u32(cp).unwrap().to_lowercase() {
            out.push(l);
        }
        i + width(cp)
    }
}

/// Lowercase — byte-exact with `chars().flat_map(char::to_lowercase)`; borrows when already lower.
pub(crate) fn lowercase<const SIMD: bool>(input: &str) -> Cow<'_, str> {
    run::<true, 0, SIMD, _>(&Lower, input)
}

// ── StripAccents ──────────────────────────────────────────────────────────────────────────────────

struct StripAcc;
impl Rule for StripAcc {
    fn scan(&self) -> &'static Scan {
        static S: OnceLock<Scan> = OnceLock::new();
        S.get_or_init(|| Scan::build(&[&SCAN_M], P_M))
    }
    fn fixup(&self, _bytes: &[u8], i: usize, cp: u32, _out: &mut String) -> usize {
        i + width(cp) // combining mark: dropped
    }
}

/// Remove combining marks (general category M) — the tk `StripAccents` predicate; no decomposition.
pub(crate) fn strip_accents<const SIMD: bool>(input: &str) -> Cow<'_, str> {
    run::<false, 0, SIMD, _>(&StripAcc, input)
}

// ── Nmt ───────────────────────────────────────────────────────────────────────────────────────────

struct Nmt;
impl Rule for Nmt {
    fn scan(&self) -> &'static Scan {
        static S: OnceLock<Scan> = OnceLock::new();
        S.get_or_init(|| Scan::build(&[&SCAN_NMT_RM, &SCAN_NMT_WS], 0))
    }
    fn fixup(&self, _bytes: &[u8], i: usize, cp: u32, out: &mut String) -> usize {
        let rm = matches!(cp,
            0x0001..=0x0008 | 0x000B | 0x000E..=0x001F | 0x007F | 0x008F | 0x009F);
        if !rm {
            out.push(' '); // the fold set (incl. the ASCII check-mode first hit)
        }
        i + width(cp)
    }
}

/// The NMT normalizer: drop its control set, fold its whitespace set to `' '`.
pub(crate) fn nmt<const SIMD: bool>(input: &str) -> Cow<'_, str> {
    run::<false, 2, SIMD, _>(&Nmt, input)
}

// ── Bert (fused clean_text + handle_chinese_chars + strip_accents + lowercase) ────────────────────

struct BertRule {
    clean: bool,
    chinese: bool,
    strip: bool,
    lower: bool,
    scan: &'static Scan,
}

impl BertRule {
    #[inline]
    fn push_lower(&self, c: char, out: &mut String) {
        if self.lower && has(&SCAN_UPPER, P_UPPER, c as u32) {
            for l in c.to_lowercase() {
                out.push(l);
            }
        } else {
            out.push(c);
        }
    }
    /// Strip+lower one isolated char: NFD it (canonical order is per-char here), drop Mn, lowercase.
    #[inline]
    fn push_stripped(&self, c: char, out: &mut String) {
        if self.strip && has(&SCAN_STRIP, P_STRIP, c as u32) {
            crate::norm::nfd_char(c, |d| {
                if !has(&SCAN_MN, P_MN, d as u32) {
                    self.push_lower(d, out);
                }
            });
        } else {
            self.push_lower(c, out);
        }
    }
}

impl Rule for BertRule {
    fn scan(&self) -> &'static Scan {
        self.scan
    }
    fn fixup(&self, bytes: &[u8], i: usize, cp: u32, out: &mut String) -> usize {
        let n = bytes.len();
        if cp < 0x80 {
            // ASCII lane stop: a clean removal / the check-mode first fold or uppercase
            let b = cp as u8;
            if self.clean {
                if matches!(b, 9 | 10 | 13) {
                    out.push(' ');
                    return i + 1;
                }
                if b < 0x20 || b == 0x7F {
                    return i + 1;
                }
            }
            debug_assert!(self.lower && b.is_ascii_uppercase());
            out.push((b | 0x20) as char);
            return i + 1;
        }
        let w = width(cp);
        let c = char::from_u32(cp).unwrap();
        if self.clean {
            if has(&SCAN_CLEAN_RM, P_CLEAN, cp) {
                return i + w; // removed before any other stage sees it
            }
            if has(&SCAN_WS, P_WS, cp) {
                out.push(' ');
                return i + w;
            }
        }
        if self.chinese && has(&SCAN_CJK, P_CJK, cp) {
            out.push(' ');
            self.push_stripped(c, out);
            out.push(' ');
            return i + w;
        }
        if self.strip && has(&SCAN_STRIP, P_STRIP, cp) {
            // the NFD cluster: following strip-relevant chars reorder as one unit, and clean-removed
            // chars are transparent to it (legacy removes them BEFORE the nfd pass)
            let (mut end, mut count, mut removed) = (i + w, 1u32, false);
            while end < n && bytes[end] >= 0x80 {
                let (c2, w2) = decode_cp(bytes, end);
                if self.chinese && has(&SCAN_CJK, P_CJK, c2) {
                    break; // padding splits the stream before nfd runs
                }
                if has(&SCAN_STRIP, P_STRIP, c2) {
                    count += 1;
                } else if self.clean && has(&SCAN_CLEAN_RM, P_CLEAN, c2) {
                    removed = true;
                } else {
                    break;
                }
                end += w2;
            }
            if count == 1 {
                self.push_stripped(c, out);
                return end;
            }
            // multi-char cluster: exact NFD (incl. cross-char reorder) via the form machinery
            let survivors: Cow<str> = if removed {
                let mut t = String::with_capacity(end - i);
                let (mut p, s) = (i, unsafe { std::str::from_utf8_unchecked(&bytes[..end]) });
                while p < end {
                    let (c2, w2) = decode_cp(bytes, p);
                    if !(self.clean && has(&SCAN_CLEAN_RM, P_CLEAN, c2)) {
                        t.push_str(&s[p..p + w2]);
                    }
                    p += w2;
                }
                Cow::Owned(t)
            } else {
                // SAFETY: [i, end) covers whole chars of valid UTF-8
                Cow::Borrowed(unsafe { std::str::from_utf8_unchecked(&bytes[i..end]) })
            };
            for d in crate::norm::decompose::<false, false>(&survivors).chars() {
                if !has(&SCAN_MN, P_MN, d as u32) {
                    self.push_lower(d, out);
                }
            }
            return end;
        }
        self.push_lower(c, out);
        i + w
    }
}

fn bert_scan(clean: bool, chinese: bool, strip: bool, lower: bool) -> &'static Scan {
    static CACHE: [OnceLock<Scan>; 16] = [const { OnceLock::new() }; 16];
    let k =
        (clean as usize) | (chinese as usize) << 1 | (strip as usize) << 2 | (lower as usize) << 3;
    CACHE[k].get_or_init(|| {
        let mut sets: Vec<&[u64; 1024]> = Vec::new();
        let mut astral = 0u8;
        if clean {
            sets.push(&SCAN_CLEAN_RM);
            sets.push(&SCAN_WS);
            astral |= P_CLEAN | P_WS;
        }
        if chinese {
            sets.push(&SCAN_CJK);
            astral |= P_CJK;
        }
        if strip {
            sets.push(&SCAN_STRIP);
            astral |= P_STRIP;
        }
        if lower {
            sets.push(&SCAN_UPPER);
            astral |= P_UPPER;
        }
        Scan::build(&sets, astral)
    })
}

/// The fused BERT normalizer (resolve `strip_accents = strip_accents.unwrap_or(lowercase)` first).
pub(crate) fn bert<const SIMD: bool>(
    input: &str,
    clean_text: bool,
    handle_chinese_chars: bool,
    strip_accents: bool,
    lowercase: bool,
) -> Cow<'_, str> {
    let r = BertRule {
        clean: clean_text,
        chinese: handle_chinese_chars,
        strip: strip_accents,
        lower: lowercase,
        scan: bert_scan(clean_text, handle_chinese_chars, strip_accents, lowercase),
    };
    match (lowercase, clean_text) {
        (true, true) => run::<true, 1, SIMD, _>(&r, input),
        (true, false) => run::<true, 0, SIMD, _>(&r, input),
        (false, true) => run::<false, 1, SIMD, _>(&r, input),
        (false, false) => run::<false, 0, SIMD, _>(&r, input),
    }
}
