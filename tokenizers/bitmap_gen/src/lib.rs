//! bitmap_gen — generator for atomsplit's shared classify tables (TAG_CLASSIFY_SPEC.md §1/§7).
//!
//! Dev-time only (depends on `unicode_categories`); nothing here is linked into the runtime crate.
//! `cargo run -p bitmap_gen` calls [`generate_atom_tables`] and writes the committed
//! `atomsplit/src/atom_tables.rs`. It bakes the dense `Tables` layout (ascii / 2-byte group / 3-byte
//! fast3 / bmp_rle / astral) read by BOTH the SIMD kernel (`vqtbl`) and the scalar reader
//! (`Tables::classify_char`); the per-codepoint value is an `Atom` (u4, stored `u8`).
use std::fmt::Write as _;
use unicode_categories::UnicodeCategories;

// Values returned here correspond to src/classify.rs
// reference atom, straight from TAG_CLASSIFY_SPEC.md §1
fn atom(c: char) -> u8 {
    let cp = c as u32;
    if c.is_letter() {
        // low nibble = Letter(0); high nibble = o200k case refinement (1=Lu∪Lt, 2=Ll, 0=caseless Lm∪Lo).
        // Coarse consumers mask it off (`in_mask` / SIMD `& 0x0F`); only `fsm_o200k` reads the nibble.
        return if c.is_letter_uppercase() || c.is_letter_titlecase() {
            0x10
        } else if c.is_letter_lowercase() {
            0x20
        } else {
            0
        };
    }
    if c.is_number() {
        return if c.is_number_decimal_digit() || c.is_number_letter() {
            1
        } else {
            2
        };
    }
    if cp == 0x0A || cp == 0x0D {
        return 3;
    }
    if cp == 0x20 {
        return 4;
    }
    if c.is_whitespace() {
        return 5;
    }
    if c.is_mark() || cp == 0x200C || cp == 0x200D {
        return 6; // \p{M} ∪ {ZWJ,ZWNJ}: real marks — ∈ deepseek/o200k `[\p{L}\p{M}]` and ∈ `\w`
    }
    if c.is_alphabetic() {
        // Other_Alphabetic non-mark (circled letters Ⓘ …, category So): a `\w` word char (coarse Mark →
        // WORD mask sees it) but NOT `[\p{L}\p{M}]`. High nibble = refine::ALPHA_SYM(1) tells deepseek /
        // o200k to treat it as the `\p{S}` symbol it categorically is, not a letter/mark.
        return 6 | (1 << 4); // 0x16: coarse Mark, refine ALPHA_SYM
    }
    if c.is_punctuation_connector() {
        return 7;
    }
    if cp == 0x27 {
        return 9;
    }
    if c.is_punctuation() {
        return 8;
    }
    if cp < 0x80 && c.is_ascii_punctuation() {
        return 8;
    }
    if c.is_numeric() {
        return 11;
    }
    if c.is_symbol() {
        return 10;
    }
    12
}

// codepoint encoded by `bytes` (synthetic UTF-8 built by the table loops). The caller's classifier
// maps it (surrogates handled there via `char::from_u32`).
fn cp_of(bytes: &[u8]) -> u32 {
    match bytes.len() {
        1 => bytes[0] as u32,
        // extract the utf8 headers to reconstruct the u32
        2 => ((bytes[0] as u32 & 0x1F) << 6) | (bytes[1] as u32 & 0x3F),
        _ => {
            ((bytes[0] as u32 & 0x0F) << 12)
                | ((bytes[1] as u32 & 0x3F) << 6)
                | (bytes[2] as u32 & 0x3F)
        }
    }
}

/// Build the dense tables from a per-codepoint `classify`, self-validate the scalar reader reproduces
/// `classify(cp)` for all 1.1M codepoints (panics → fails the build on any mismatch), and return the
/// Rust source for `pub static {struct_name}: Tables` (the committed `atom_tables.rs`). `classify` is the
/// single source of truth; every table (and the SIMD kernel) is derived from it. `kind` labels the
/// emitted doc header.
fn generate_tables(struct_name: &str, kind: &str, classify: &dyn Fn(u32) -> u8) -> String {
    let tag = |bytes: &[u8]| classify(cp_of(bytes));

    // ASCII 128-entry table (two halves for the subtract-trick lookup)
    let mut ascii_lo = [0u8; 64];
    let mut ascii_hi = [0u8; 64];
    for b in 0..64u8 {
        ascii_lo[b as usize] = tag(&[b]);
        ascii_hi[b as usize] = tag(&[64 + b]);
    }

    // 2-byte: 8 groups × 4 leads × 64 conts
    let mut group = [[[0u8; 64]; 4]; 8];
    for g in 0..8usize {
        // 0xC0 is 110..... the utf8 header
        // g is the group: 110xxx.. *4 shifts it by 2
        let base = 0xC0u8 + (g as u8) * 4;
        for k in 0..4usize {
            for c in 0..64u8 {
                // 0x80 is 10yyyyyy, its constructing the 2nd byte.
                // We add `k` as its the sub group (0,1,2,3)
                group[g][k][c as usize] = tag(&[base + k as u8, 0x80 | c]);
            }
        }
    }
    // Finally we build the hardest of them all:
    let mut fast3_uni = [0u8; 512];
    let mut fast3_slot = [0u16; 512];
    let mut fast3_mixed: Vec<([u8; 64], [u8; 64])> = Vec::new();
    for lead in 0xE0u8..=0xEF {
        for p in 0..32u8 {
            let (b2e, b2o) = (0x80 + 2 * p, 0x80 + 2 * p + 1);
            let (mut lo, mut hi) = ([0u8; 64], [0u8; 64]);
            for c in 0..64u8 {
                lo[c as usize] = tag(&[lead, b2e, 0x80 | c]);
                hi[c as usize] = tag(&[lead, b2o, 0x80 | c]);
            }
            let idx = (lead - 0xE0) as usize * 32 + p as usize;
            if lo.iter().chain(hi.iter()).all(|&x| x == lo[0]) {
                fast3_uni[idx] = lo[0];
            } else {
                fast3_uni[idx] = 0xFF;
                fast3_slot[idx] = fast3_mixed.len() as u16;
                fast3_mixed.push((lo, hi));
            }
        }
    }

    // Cold fallback: dense BMP tag → run-length encode (emit only on tag change)
    let mut bmp_rle: Vec<(u16, u8)> = Vec::new();
    for cp in 0..0x10000u32 {
        let a = classify(cp);
        if bmp_rle.last().map_or(true, |&(_, la)| la != a) {
            bmp_rle.push((cp as u16, a));
        }
    }

    // astral (cp >= 0x10000), run-length encoded (start_cp, tag).
    let mut astral: Vec<(u32, u8)> = Vec::new();
    for cp in 0x10000u32..=0x10FFFF {
        let a = classify(cp);
        if astral.last().map_or(true, |&(_, la)| la != a) {
            astral.push((cp, a));
        }
    }

    // ── self-validation: the scalar reader (as `Tables::classify_char` indexes these) == classify(cp) ──
    let read = |cp: u32| -> u8 {
        if cp < 0x80 {
            return if cp < 64 {
                ascii_lo[cp as usize]
            } else {
                ascii_hi[(cp - 64) as usize]
            };
        }
        if cp < 0x800 {
            let b0 = 0xC0 | (cp >> 6) as u8;
            return group[((b0 >> 2) & 7) as usize][(b0 & 3) as usize][(cp & 0x3F) as usize];
        }
        if cp < 0x10000 {
            let b0 = 0xE0 | (cp >> 12) as u8;
            let b1 = ((cp >> 6) & 0x3F) as u8;
            let block = (b0 - 0xE0) as usize * 32 + ((b1 >> 1) & 0x1F) as usize;
            let uni = fast3_uni[block];
            if uni != 0xFF {
                return uni;
            }
            let (lo, hi) = &fast3_mixed[fast3_slot[block] as usize];
            return if b1 & 1 == 0 {
                lo[(cp & 0x3F) as usize]
            } else {
                hi[(cp & 0x3F) as usize]
            };
        }
        astral[astral.partition_point(|&(s, _)| s <= cp) - 1].1
    };
    for cp in 0..=0x10FFFFu32 {
        if (0xD800..=0xDFFF).contains(&cp) {
            continue;
        }
        assert_eq!(
            read(cp),
            classify(cp),
            "bitmap_gen: table mismatch at cp={cp:#06x}"
        );
    }

    // ── emit the Rust source ──
    let mut o = String::new();
    o.push_str(
        "//! GENERATED by `bitmap_gen` — do NOT edit. Regenerate with `cargo run -p bitmap_gen`.\n",
    );
    writeln!(o, "//! Dense {kind} classify tables shared by the SIMD kernel (`vqtbl`) and the scalar reader").unwrap();
    o.push_str("//! `Tables::classify_char`. `bitmap_gen` self-validates all 1.1M codepoints. Spec §1/§7.\n");
    o.push_str("use crate::simd_classify::Tables;\n\n");

    let emit_u8 = |o: &mut String, name: &str, t: &[u8]| {
        write!(o, "#[rustfmt::skip]\nstatic {name}: [u8; {}] = [", t.len()).unwrap();
        for (i, v) in t.iter().enumerate() {
            if i > 0 {
                o.push(',');
            }
            write!(o, "{v}").unwrap();
        }
        o.push_str("];\n");
    };

    o.push_str("#[rustfmt::skip]\nstatic GROUP: [[[u8; 64]; 4]; 8] = [");
    for g in 0..8 {
        o.push('[');
        for k in 0..4 {
            o.push('[');
            for c in 0..64 {
                if c > 0 {
                    o.push(',');
                }
                write!(o, "{}", group[g][k][c]).unwrap();
            }
            o.push(']');
            if k < 3 {
                o.push(',');
            }
        }
        o.push(']');
        if g < 7 {
            o.push(',');
        }
    }
    o.push_str("];\n");

    emit_u8(&mut o, "ASCII_LO", &ascii_lo);
    emit_u8(&mut o, "ASCII_HI", &ascii_hi);
    emit_u8(&mut o, "FAST3_UNI", &fast3_uni);

    o.push_str("#[rustfmt::skip]\nstatic FAST3_SLOT: [u16; 512] = [");
    for (i, v) in fast3_slot.iter().enumerate() {
        if i > 0 {
            o.push(',');
        }
        write!(o, "{v}").unwrap();
    }
    o.push_str("];\n");

    write!(
        o,
        "#[rustfmt::skip]\nstatic FAST3_MIXED: [([u8; 64], [u8; 64]); {}] = [",
        fast3_mixed.len()
    )
    .unwrap();
    for (i, (lo, hi)) in fast3_mixed.iter().enumerate() {
        if i > 0 {
            o.push(',');
        }
        o.push('(');
        for (half, arr) in [lo, hi].iter().enumerate() {
            o.push('[');
            for (j, v) in arr.iter().enumerate() {
                if j > 0 {
                    o.push(',');
                }
                write!(o, "{v}").unwrap();
            }
            o.push(']');
            if half == 0 {
                o.push(',');
            }
        }
        o.push(')');
    }
    o.push_str("];\n");

    write!(
        o,
        "#[rustfmt::skip]\nstatic BMP_RLE: [(u16, u8); {}] = [",
        bmp_rle.len()
    )
    .unwrap();
    for (i, (s, a)) in bmp_rle.iter().enumerate() {
        if i > 0 {
            o.push(',');
        }
        write!(o, "(0x{s:X},{a})").unwrap();
    }
    o.push_str("];\n");

    write!(
        o,
        "#[rustfmt::skip]\nstatic ASTRAL: [(u32, u8); {}] = [",
        astral.len()
    )
    .unwrap();
    for (i, (s, a)) in astral.iter().enumerate() {
        if i > 0 {
            o.push(',');
        }
        write!(o, "(0x{s:X},{a})").unwrap();
    }
    o.push_str("];\n\n");

    write!(o, "pub static {struct_name}: Tables = Tables {{\n").unwrap();
    o.push_str("    ascii_lo: ASCII_LO,\n    ascii_hi: ASCII_HI,\n    group_tables: GROUP,\n");
    o.push_str("    fast3_uni: FAST3_UNI,\n    fast3_slot: FAST3_SLOT,\n");
    o.push_str(
        "    fast3_mixed: &FAST3_MIXED,\n    bmp_rle: &BMP_RLE,\n    astral: &ASTRAL,\n};\n",
    );
    o
}

/// Atom tables — the 12-way category alphabet (`ATOM_TABLES`), with the `Letter` coarse class carrying an
/// o200k case refinement in its high nibble (see [`atom`]).
pub fn generate_atom_tables() -> String {
    generate_tables("ATOM_TABLES", "atom", &|cp| {
        char::from_u32(cp).map(atom).unwrap_or(10)
    })
}
