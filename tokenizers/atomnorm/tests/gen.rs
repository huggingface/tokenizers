//! Table generator: derives EVERYTHING from `unicode-normalization` (the byte-exactness oracle) and
//! writes `src/tables.rs`. Run: `cargo test -p atomnorm --release generate -- --ignored`
use std::collections::{HashMap, HashSet};
use std::fmt::Write;
use unicode_normalization::UnicodeNormalization;
use unicode_normalization::char::{
    canonical_combining_class as ccc, compose, decompose_canonical, decompose_compatible,
};

const HANGUL: std::ops::RangeInclusive<u32> = 0xAC00..=0xD7A3;

fn full_d(cp: u32, compat: bool) -> Vec<char> {
    let c = char::from_u32(cp).unwrap();
    let mut v = Vec::new();
    if compat {
        decompose_compatible(c, |d| v.push(d));
    } else {
        decompose_canonical(c, |d| v.push(d));
    }
    v
}
fn changes(cp: u32, compat: bool) -> bool {
    let d = full_d(cp, compat);
    d.len() != 1 || d[0] as u32 != cp
}

struct Gen {
    rank: HashMap<u8, u8>,
    maybe: HashSet<u32>,
    pairs: Vec<(u64, u32)>,
    tag: Vec<u8>, // 0x110000 entries
}

impl Gen {
    fn build() -> Self {
        // ccc → order-preserving rank
        let mut cccs: Vec<u8> = (0..0x110000u32)
            .filter_map(char::from_u32)
            .map(ccc)
            .filter(|&c| c != 0)
            .collect();
        cccs.sort_unstable();
        cccs.dedup();
        assert!(cccs.len() <= 0x3B, "rank overflow: {}", cccs.len());
        let rank: HashMap<u8, u8> = cccs
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, i as u8 + 1))
            .collect();

        // primary composites: recover the raw pair of each canonically-decomposing char by
        // recomposing all-but-last of its full decomposition, then verifying with `compose`.
        let mut pairs: Vec<(u64, u32)> = Vec::new();
        let mut maybe: HashSet<u32> = (0x1161..=0x1175).chain(0x11A8..=0x11C2).collect(); // V/T jamo
        for cp in 0..0x110000u32 {
            let Some(_) = char::from_u32(cp) else {
                continue;
            };
            if HANGUL.contains(&cp) || !changes(cp, false) {
                continue;
            }
            let d = full_d(cp, false);
            if d.len() < 2 {
                continue; // singleton decomposition: never a composition target
            }
            let head: String = d[..d.len() - 1].iter().collect();
            let a: Vec<char> = head.nfc().collect();
            if a.len() != 1 {
                continue;
            }
            let b = d[d.len() - 1];
            if compose(a[0], b) == Some(char::from_u32(cp).unwrap()) {
                pairs.push((((a[0] as u64) << 21) | b as u64, cp));
                maybe.insert(b as u32);
            }
        }
        pairs.sort_unstable();

        // per-cp tag
        let mut tag = vec![0u8; 0x110000];
        for cp in 0..0x110000u32 {
            let Some(c) = char::from_u32(cp) else {
                continue;
            };
            let t = &mut tag[cp as usize];
            let r = ccc(c).checked_sub(1).map_or(0, |_| rank[&ccc(c)]);
            if changes(cp, false) {
                // 0x7E only when the char is stable under BOTH composed forms — else compose must window
                let orig = c.to_string();
                let stable = orig.nfc().collect::<String>() == orig
                    && orig.nfkc().collect::<String>() == orig;
                *t = if stable { 0x7E } else { 0x7D };
            } else if changes(cp, true) {
                *t = if r == 0 { 0x3C } else { 0x3D };
            } else if maybe.contains(&cp) {
                *t = 0x40 | r;
            } else {
                *t = r;
            }
        }
        Gen {
            rank,
            maybe,
            pairs,
            tag,
        }
    }

    /// Returns `(idx, data_len)` so the NFKC parallel table can reuse the same slot mapping.
    fn blobs(&self, compat: bool, o: &mut String, name: &str) -> (Vec<u32>, usize) {
        // trie: IDX[cp>>6] → 64-slot base in DATA; blob bytes with [first_rank, last_rank, mark_off]
        let (mut data, mut blob): (Vec<u32>, Vec<u8>) = (vec![0; 64], Vec::new()); // block 0 = all-absent
        let mut idx = vec![0u32; 0x30000 >> 6];
        for (blk, ix) in idx.iter_mut().enumerate() {
            let mut slots = [0u32; 64];
            let mut any = false;
            for lo in 0..64u32 {
                let cp = ((blk as u32) << 6) | lo;
                if char::from_u32(cp).is_none() || HANGUL.contains(&cp) || !changes(cp, compat) {
                    continue;
                }
                let d = full_d(cp, compat);
                let (mut bytes, mut mark_off, mut prev, mut sorted) =
                    (Vec::new(), 0usize, 0u8, true);
                for &ch in &d {
                    let r = if ccc(ch) == 0 { 0 } else { self.rank[&ccc(ch)] };
                    if r != 0 && r < prev {
                        sorted = false;
                    }
                    let mut buf = [0u8; 4];
                    bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                    if r == 0 {
                        mark_off = bytes.len();
                    }
                    prev = r;
                }
                assert!(
                    sorted && bytes.len() <= 0xFE,
                    "U+{cp:04X}: unsupported decomposition"
                );
                let first = if ccc(d[0]) == 0 {
                    0
                } else {
                    self.rank[&ccc(d[0])]
                };
                let last = if ccc(*d.last().unwrap()) == 0 {
                    0
                } else {
                    self.rank[&ccc(*d.last().unwrap())]
                };
                blob.extend_from_slice(&[first, last, mark_off as u8]);
                let off = blob.len() as u32;
                blob.extend_from_slice(&bytes);
                slots[lo as usize] = (off << 8) | bytes.len() as u32;
                any = true;
            }
            if any {
                *ix = data.len() as u32;
                data.extend_from_slice(&slots);
            }
        }
        blob.extend_from_slice(&[0; 16]);
        emit_u32(o, &format!("{name}_IDX"), &idx);
        let data_len = data.len();
        emit_u32(o, &format!("{name}_DATA"), &data);
        emit_u8(o, &format!("{name}_BLOB"), &blob);
        (idx, data_len)
    }

    fn nfkc_blobs(
        &self,
        o: &mut String,
        nfkd_data_len: usize,
        nfkd_idx: &dyn Fn(u32) -> Option<usize>,
    ) {
        // composed blobs parallel to NFKD_DATA, only for neighbour-inert compat starters (tag 0x3C)
        let firsts: HashSet<u32> = self
            .pairs
            .iter()
            .map(|&(k, _)| (k >> 21) as u32)
            .chain(0x1100..=0x1112)
            .chain((0xAC00..=0xD7A3).step_by(28))
            .collect();
        let mut data = vec![0u32; nfkd_data_len];
        let mut blob: Vec<u8> = Vec::new();
        for cp in 0..0x30000u32 {
            let Some(c) = char::from_u32(cp) else {
                continue;
            };
            if self.tag[cp as usize] != 0x3C {
                continue;
            }
            let Some(slot) = nfkd_idx(cp) else { continue };
            let k: String = c.to_string().nfkc().collect();
            let chars: Vec<char> = k.chars().collect();
            let safe = !k.is_empty()
                && k.len() <= 16
                && chars
                    .iter()
                    .all(|&ch| ccc(ch) == 0 && !changes(ch as u32, false))
                && !self.maybe.contains(&(chars[0] as u32))
                && !firsts.contains(&(*chars.last().unwrap() as u32));
            if !safe {
                continue;
            }
            let off = blob.len() as u32;
            blob.extend_from_slice(k.as_bytes());
            data[slot] = ((off + 1) << 8) | k.len() as u32; // +1 keeps off != 0 unambiguous
        }
        blob.extend_from_slice(&[0; 16]);
        // shift offsets back by 1 at runtime? simpler: prepend one pad byte so off+1 indexes correctly
        let mut blob2 = vec![0u8];
        blob2.extend_from_slice(&blob);
        emit_u32(o, "NFKC_DATA", &data);
        emit_u8(o, "NFKC_BLOB", &blob2);
    }
}

fn emit_u64(o: &mut String, name: &str, v: &[u64]) {
    write!(
        o,
        "#[rustfmt::skip]\npub static {name}: [u64; {}] = [",
        v.len()
    )
    .unwrap();
    for (i, x) in v.iter().enumerate() {
        if i > 0 {
            o.push(',');
        }
        write!(o, "{x}").unwrap();
    }
    o.push_str("];\n");
}

fn emit_u8(o: &mut String, name: &str, v: &[u8]) {
    write!(
        o,
        "#[rustfmt::skip]\npub static {name}: [u8; {}] = [",
        v.len()
    )
    .unwrap();
    for (i, x) in v.iter().enumerate() {
        if i > 0 {
            o.push(',');
        }
        write!(o, "{x}").unwrap();
    }
    o.push_str("];\n");
}
fn emit_u32(o: &mut String, name: &str, v: &[u32]) {
    write!(
        o,
        "#[rustfmt::skip]\npub static {name}: [u32; {}] = [",
        v.len()
    )
    .unwrap();
    for (i, x) in v.iter().enumerate() {
        if i > 0 {
            o.push(',');
        }
        write!(o, "{x}").unwrap();
    }
    o.push_str("];\n");
}

#[test]
#[ignore = "writes src/tables.rs — run explicitly to regenerate"]
fn generate() {
    let g = Gen::build();
    let t = &g.tag;
    // relevance per form, for LEAD_SUSPECT
    let d_rel = |t: u8, k: bool| {
        t >= 0x7D
            || (t & 0x3F >= 1 && t & 0x3F <= 0x3B)
            || t & 0x3F == 0x3D
            || (k && t & 0x3F == 0x3C)
    };
    let c_rel = |t: u8, k: bool| {
        (t & 0x40 != 0 && t < 0x7D)
            || t == 0x7D
            || t & 0x3F == 0x3D
            || (t & 0x3F >= 1 && t & 0x3F <= 0x3B)
            || (k && t & 0x3F == 0x3C)
    };
    let mut lead = [0u8; 64];
    let mut astral_b1 = 0u64;
    for cp in 0x80..0x110000u32 {
        let Some(c) = char::from_u32(cp) else {
            continue;
        };
        let tg = t[cp as usize];
        if tg == 0 {
            continue;
        }
        let mut buf = [0u8; 4];
        let enc = c.encode_utf8(&mut buf).as_bytes();
        let li = (enc[0] - 0xC0) as usize;
        for (bit, rel) in [
            (1u8, d_rel(tg, false)),
            (2, d_rel(tg, true)),
            (4, c_rel(tg, false)),
            (8, c_rel(tg, true)),
        ] {
            if rel {
                lead[li] |= bit;
            }
        }
        if enc.len() == 4
            && (d_rel(tg, true) || c_rel(tg, true) || d_rel(tg, false) || c_rel(tg, false))
        {
            astral_b1 |= 1 << (enc[1] & 0x3F);
        }
    }
    // BMP union bitmap + astral RLE
    let mut bmp = vec![0u64; 1024];
    for cp in 0..0x10000u32 {
        if t[cp as usize] != 0 {
            bmp[(cp >> 6) as usize] |= 1 << (cp & 63);
        }
    }
    let mut astral: Vec<(u32, u8)> = vec![(0x10000, t[0x10000])];
    for cp in 0x10001..0x110000u32 {
        if t[cp as usize] != astral.last().unwrap().1 {
            astral.push((cp, t[cp as usize]));
        }
    }

    let mut o = String::from(
        "//! GENERATED — do NOT edit. Regenerate: `cargo test -p atomnorm --release generate -- --ignored`.\n\
         //! Derived from `unicode-normalization`; see `lib.rs` for the tag encoding & layouts.\n\n",
    );
    emit_u8(&mut o, "TAG", &t[..0x10000]);
    // per-form 0/FF masks (vqtbl output IS the hit mask — no vtst in the layer-0 loop)
    for (name, bit) in [
        ("LEAD_NFD", 1u8),
        ("LEAD_NFKD", 2),
        ("LEAD_NFC", 4),
        ("LEAD_NFKC", 8),
    ] {
        let m: Vec<u8> = lead
            .iter()
            .map(|&l| if l & bit != 0 { 0xFF } else { 0 })
            .collect();
        emit_u8(&mut o, name, &m);
    }
    {
        write!(o, "#[rustfmt::skip]\npub static BMP_SET: [u64; 1024] = [").unwrap();
        for (i, x) in bmp.iter().enumerate() {
            if i > 0 {
                o.push(',');
            }
            write!(o, "{x}").unwrap();
        }
        o.push_str("];\n");
        write!(
            o,
            "#[rustfmt::skip]\npub static ASTRAL: [(u32, u8); {}] = [",
            astral.len()
        )
        .unwrap();
        for (i, (s, v)) in astral.iter().enumerate() {
            if i > 0 {
                o.push(',');
            }
            write!(o, "({s},{v})").unwrap();
        }
        o.push_str("];\n");
        writeln!(o, "pub static ASTRAL_B1: u64 = {astral_b1};").unwrap();
    }
    g.blobs(false, &mut o, "NFD");
    // the NFKC parallel table reuses NFKD's slot mapping
    let (nfkd_idx, nfkd_data_len) = g.blobs(true, &mut o, "NFKD");
    g.nfkc_blobs(&mut o, nfkd_data_len, &|cp| {
        let base = nfkd_idx[(cp >> 6) as usize];
        if base == 0 {
            None
        } else {
            Some((base + (cp & 63)) as usize)
        }
    });
    {
        write!(
            o,
            "#[rustfmt::skip]\npub static COMPOSE: [(u64, u32); {}] = [",
            g.pairs.len()
        )
        .unwrap();
        for (i, (k, v)) in g.pairs.iter().enumerate() {
            if i > 0 {
                o.push(',');
            }
            write!(o, "({k},{v})").unwrap();
        }
        o.push_str("];\n");
    }
    // ── scan property sets (Lowercase / StripAccents / Nmt / Bert) ────────────────────────────────
    // Derived from the SAME predicates the tk-encode legacy normalizers run (std `to_lowercase`,
    // `unicode_categories`, `unicode_normalization_alignments`), so the baked sets are bug-compatible.
    {
        use unicode_categories::UnicodeCategories;
        let lowercases_to_self = |c: char| {
            let mut it = c.to_lowercase();
            matches!((it.next(), it.next()), (Some(first), None) if first == c)
        };
        let bert_ws = |c: char| matches!(c, '\t' | '\n' | '\r') || c.is_whitespace();
        let bert_ctrl = |c: char| !matches!(c, '\t' | '\n' | '\r') && c.is_other();
        let cjk = |cp: u32| {
            matches!(cp,
                0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0x20000..=0x2A6DF | 0x2A700..=0x2B73F |
                0x2B740..=0x2B81F | 0x2B920..=0x2CEAF | 0xF900..=0xFAFF | 0x2F800..=0x2FA1F)
        };
        let nmt_rm = |cp: u32| matches!(cp, 0x0001..=0x0008 | 0x000B | 0x000E..=0x001F | 0x007F | 0x008F | 0x009F);
        let nmt_ws = |cp: u32| {
            matches!(
                cp,
                0x0009 | 0x000A | 0x000C | 0x000D | 0x1680 | 0x200B
                    ..=0x200F | 0x2028 | 0x2029 | 0x2581 | 0xFEFF | 0xFFFD
            )
        };
        // one bit-per-property function; bits are `scan.rs`'s P_* constants
        let props = |cp: u32| -> u16 {
            let Some(c) = char::from_u32(cp) else {
                return 0;
            };
            let mut p = 0u16;
            if !lowercases_to_self(c) {
                p |= 1; // P_UPPER
            }
            if c.is_mark_nonspacing() {
                p |= 2; // P_MN (bert strip filter)
            }
            if unicode_normalization_alignments::char::is_combining_mark(c) {
                p |= 4; // P_M (StripAccents)
            }
            if c == '\0' || c == '\u{fffd}' || bert_ctrl(c) {
                p |= 8; // P_CLEAN (bert clean_text removes)
            }
            if cjk(cp) {
                p |= 16; // P_CJK
            }
            if changes(cp, false) || ccc(c) != 0 || c.is_mark_nonspacing() {
                p |= 32; // P_STRIP (NFD-affected ∪ marks: bert strip_accents relevance)
            }
            if c != ' ' && bert_ws(c) {
                p |= 64; // P_WS (bert clean_text folds to ' ')
            }
            if nmt_rm(cp) {
                p |= 128; // P_NMT_RM
            }
            if nmt_ws(cp) {
                p |= 256; // P_NMT_WS
            }
            p
        };
        for (name, bit) in [
            ("SCAN_UPPER", 1u16),
            ("SCAN_MN", 2),
            ("SCAN_M", 4),
            ("SCAN_CLEAN_RM", 8),
            ("SCAN_CJK", 16),
            ("SCAN_STRIP", 32),
            ("SCAN_WS", 64),
            ("SCAN_NMT_RM", 128),
            ("SCAN_NMT_WS", 256),
        ] {
            let mut bmp = vec![0u64; 1024];
            for cp in 0..0x10000u32 {
                if props(cp) & bit != 0 {
                    bmp[(cp >> 6) as usize] |= 1 << (cp & 63);
                }
            }
            emit_u64(&mut o, name, &bmp);
        }
        // the "regular case" 2-byte set: uppercase whose lowercase is exactly cp + 0x20 — the
        // source table of the in-lane two-table case swap (target = arithmetic, +0x20 with a
        // UTF-8 carry). Everything else cased stays a scalar fixup.
        let mut reg2 = vec![0u64; 32];
        for cp in 0x80..0x800u32 {
            let c = char::from_u32(cp).unwrap();
            let mut lo = c.to_lowercase();
            if lo.next() == char::from_u32(cp + 0x20) && lo.next().is_none() {
                assert!(
                    cp + 0x20 < 0x800,
                    "regular-case mapping leaves the 2-byte range"
                );
                reg2[(cp >> 6) as usize] |= 1 << (cp & 63);
            }
        }
        emit_u64(&mut o, "SCAN_REG2", &reg2);
        // astral runs: (start, props) RLE — nmt has no astral members so u8 fits the 7 used bits
        let mut runs: Vec<(u32, u8)> = vec![(0x10000, props(0x10000) as u8)];
        for cp in 0x10001..0x110000u32 {
            assert!(props(cp) < 0x100, "astral prop overflows the u8 RLE");
            let p = props(cp) as u8;
            if p != runs.last().unwrap().1 {
                runs.push((cp, p));
            }
        }
        write!(
            o,
            "#[rustfmt::skip]\npub static SCAN_ASTRAL: [(u32, u8); {}] = [",
            runs.len()
        )
        .unwrap();
        for (i, (s, v)) in runs.iter().enumerate() {
            if i > 0 {
                o.push(',');
            }
            write!(o, "({s},{v})").unwrap();
        }
        o.push_str("];\n");
        eprintln!("scan astral RLE runs: {}", runs.len());
    }

    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/tables.rs");
    std::fs::write(path, o).unwrap();
    eprintln!("wrote {path}");
}
