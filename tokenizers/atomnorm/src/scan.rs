use crate::norm::{decode_cp, raw_extend};
use crate::tables::*;
use std::borrow::Cow;
use std::sync::OnceLock;

const P_UPPER: u8 = 1;
const P_MN: u8 = 2;
const P_M: u8 = 4;
const P_CLEAN: u8 = 8;
const P_CJK: u8 = 16;
const P_STRIP: u8 = 32;
const P_WS: u8 = 64;

#[inline]
fn astral_props(cp: u32) -> u8 {
    match SCAN_ASTRAL.binary_search_by(|&(s, _)| s.cmp(&cp)) {
        Ok(k) => SCAN_ASTRAL[k].1,
        Err(k) => SCAN_ASTRAL[k - 1].1,
    }
}

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

pub(crate) struct Scan {
    set: [u64; 1024],
    lead: [u8; 64],
    astral: u8,
    astral_all: bool,
    reg2: [u64; 32],
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
        for l in 0xC2u8..=0xDF {
            if set[(l & 0x1F) as usize] != 0 {
                lead[(l - 0xC0) as usize] = 0xFF;
            }
        }
        for l in 0xE0u8..=0xEF {
            let base = (((l & 0x0F) as usize) << 6).max(32);
            if set[base..((l & 0x0F) as usize + 1) << 6]
                .iter()
                .any(|&w| w != 0)
            {
                lead[(l - 0xC0) as usize] = 0xFF;
            }
        }
        if astral != 0 {
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
            reg2: [0; 32],
        }
    }
    fn with_case_swap(mut self, exclude: &[&[u64; 1024]]) -> Scan {
        for w in 0..32 {
            let mut x = SCAN_REG2[w];
            for e in exclude {
                x &= !e[w];
            }
            self.reg2[w] = x;
        }
        self
    }
    #[inline]
    fn hit_bmp(&self, cp: u32) -> bool {
        self.set[(cp >> 6) as usize] >> (cp & 63) & 1 != 0
    }
    #[inline]
    pub(crate) fn contains(&self, cp: u32) -> bool {
        if cp < 0x10000 {
            self.hit_bmp(cp)
        } else {
            self.astral_all || astral_props(cp) & self.astral != 0
        }
    }
    pub(crate) fn next_member<const SIMD: bool>(&self, bytes: &[u8], i: usize) -> usize {
        let mut dummy = String::new();
        next_hit::<false, false, 0, SIMD, true>(bytes, i, &mut dummy, self).0
    }
}

#[cfg_attr(
    not(target_arch = "aarch64"),
    allow(unused_variables, unused_assignments)
)]
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
    let mut streak = 0u32;
    let mut streak2 = 0u32;
    while i < n {
        #[cfg(target_arch = "aarch64")]
        if SIMD {
            if streak >= 8 {
                i = crate::simd_norm::scan_prefix::<STORE, LOWER, CLEAN, ASCII_SET>(
                    bytes, i, out, &sc.lead, &sc.set,
                );
                (streak, streak2) = (0, 0);
                if i >= n {
                    break;
                }
            } else if STORE && LOWER && streak2 >= 8 {
                i = crate::simd_norm::scan2_case::<CLEAN>(bytes, i, out, &sc.set, &sc.reg2);
                (streak, streak2) = (0, 0);
                if i >= n {
                    break;
                }
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
                unsafe { out.as_mut_vec().push(t) };
            }
            i += 1;
            streak += 1;
            streak2 += 1;
            continue;
        }
        if b < 0xC0 {
            if STORE {
                unsafe { out.as_mut_vec().push(b) };
            }
            i += 1;
            streak += 1;
            streak2 += 1;
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
        streak2 = if w == 2 { streak2 + 2 } else { 0 };
        if m != 0 {
            let (cp, _) = decode_cp(bytes, i);
            if LOWER && cp < 0x800 && sc.reg2[(cp >> 6) as usize] >> (cp & 63) & 1 != 0 {
                if !STORE {
                    return (i, cp);
                }
                let lc = cp + 0x20;
                unsafe {
                    let v = out.as_mut_vec();
                    v.push(0xC0 | (lc >> 6) as u8);
                    v.push(0x80 | (lc & 0x3F) as u8);
                }
                i += 2;
                continue;
            }
            if sc.contains(cp) {
                return (i, cp);
            }
            streak = 0;
        } else {
            streak += w as u32;
        }
        if STORE {
            unsafe { raw_extend(out, bytes.as_ptr().add(i), w, w) };
        }
        i += w;
    }
    (n, 0)
}

trait Rule {
    fn scan(&self) -> &'static Scan;
    fn fixup(&self, bytes: &[u8], i: usize, cp: u32, out: &mut String) -> usize;
}

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
        out.reserve(n - i + 32);
        let (pos, c2) = next_hit::<true, LOWER, CLEAN, SIMD, false>(bytes, i, &mut out, sc);
        if pos >= n {
            break;
        }
        (i, cp) = (pos, c2);
    }
    Cow::Owned(out)
}

struct Lower;
impl Rule for Lower {
    fn scan(&self) -> &'static Scan {
        static S: OnceLock<Scan> = OnceLock::new();
        S.get_or_init(|| Scan::build(&[&SCAN_UPPER], P_UPPER).with_case_swap(&[]))
    }
    fn fixup(&self, _bytes: &[u8], i: usize, cp: u32, out: &mut String) -> usize {
        for l in char::from_u32(cp).unwrap().to_lowercase() {
            out.push(l);
        }
        i + width(cp)
    }
}

pub(crate) fn lowercase<const SIMD: bool>(input: &str) -> Cow<'_, str> {
    run::<true, 0, SIMD, _>(&Lower, input)
}

struct StripAcc;
impl Rule for StripAcc {
    fn scan(&self) -> &'static Scan {
        static S: OnceLock<Scan> = OnceLock::new();
        S.get_or_init(|| Scan::build(&[&SCAN_M], P_M))
    }
    fn fixup(&self, _bytes: &[u8], i: usize, cp: u32, _out: &mut String) -> usize {
        i + width(cp)
    }
}

pub(crate) fn strip_accents<const SIMD: bool>(input: &str) -> Cow<'_, str> {
    run::<false, 0, SIMD, _>(&StripAcc, input)
}

struct Nmt;
impl Rule for Nmt {
    fn scan(&self) -> &'static Scan {
        static S: OnceLock<Scan> = OnceLock::new();
        S.get_or_init(|| Scan::build(&[&SCAN_NMT_RM, &SCAN_NMT_WS], 0))
    }
    fn fixup(&self, _bytes: &[u8], i: usize, cp: u32, out: &mut String) -> usize {
        if !has(&SCAN_NMT_RM, 0, cp) {
            out.push(' ');
        }
        i + width(cp)
    }
}

pub(crate) fn nmt<const SIMD: bool>(input: &str) -> Cow<'_, str> {
    run::<false, 2, SIMD, _>(&Nmt, input)
}

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
                return i + w;
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
            let (mut end, mut count, mut removed) = (i + w, 1u32, false);
            while end < n && bytes[end] >= 0x80 {
                let (c2, w2) = decode_cp(bytes, end);
                if self.chinese && has(&SCAN_CJK, P_CJK, c2) {
                    break;
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
        let mut other: Vec<&[u64; 1024]> = Vec::new();
        let mut astral = 0u8;
        if clean {
            other.push(&SCAN_CLEAN_RM);
            other.push(&SCAN_WS);
            astral |= P_CLEAN | P_WS;
        }
        if chinese {
            other.push(&SCAN_CJK);
            astral |= P_CJK;
        }
        if strip {
            other.push(&SCAN_STRIP);
            astral |= P_STRIP;
        }
        let mut sets = other.clone();
        if lower {
            sets.push(&SCAN_UPPER);
            astral |= P_UPPER;
        }
        let sc = Scan::build(&sets, astral);
        if lower { sc.with_case_swap(&other) } else { sc }
    })
}

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
