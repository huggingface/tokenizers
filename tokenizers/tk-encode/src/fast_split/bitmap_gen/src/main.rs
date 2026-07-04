/// This will store what we need to build the unicode free pretokenization tables.
/// iterates codepoints in the 2-byte (U+0080–07FF) and 3-byte (U+0800–FFFF) ranges,
/// tests membership, sets bits, and emits static LETTER2: [u64;32], LETTER3: [u64;1024],
/// NUM2/NUM3, … into a committed unicode_tables.rs.
/// Index scheme like the POC: LETTER3[((lead-0xE0)<<6)|(b2-0x80)] >> (b3&0x3F).
/// 4-byte (astral) letters are rare → a small fallback fn, not a giant table.
/// This is only used in fallback / non SIMD path.

const LETTER_RANGES: &[(u32, u32)] = &[(0x41, 0x5A), (0x61, 0x7A), (0xC0, 0xD6)]; // \p{L}, pinned to onig's Unicode ver
const fn is_letter(cp: u32) -> bool {
    todo!() /* const binary/linear search over LETTER_RANGES */
}
const LETTER2: [u64; 32] = {
    let mut t = [0u64; 32];
    let mut cp = 0x800;
    while cp <= 0xFFFF {
        if is_letter(cp) {
            todo!()
        }
        cp += 1;
    }
    t
};
const LETTER3: [u64; 1024] = {
    let mut t = [0u64; 1024];
    let mut cp = 0x800;
    while cp <= 0xFFFF {
        if is_letter(cp) {
            todo!()
        }
        cp += 1;
    }
    t
};
const NUMBER2: [u64; 32] = {
    let mut t = [0u64; 32];
    let mut cp = 0x800;
    while cp <= 0xFFFF {
        if is_letter(cp) {
            todo!()
        }
        cp += 1;
    }
    t
};
const NUMBER3: [u64; 1024] = {
    let mut t = [0u64; 1024];
    let mut cp = 0x800;
    while cp <= 0xFFFF {
        if is_letter(cp) {
            todo!()
        }
        cp += 1;
    }
    t
};
fn main() {
    println!("Generating bitmap table for fast nibble based splitting");
}
