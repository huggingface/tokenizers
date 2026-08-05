//! Which characters convert straight to one internal symbol (they "fold"), and the compact
//! table that answers it at encode time.
//!
//! For a character-level model, a character folds when the vocab has a one-character token for
//! it. For a byte-level model, a character reaches the merge loop as bytes, so folding it means
//! proving the merges are predetermined: see `byte_level`.
mod byte_level;

use ahash::AHashMap;
use byte_level::{ByteLevelFold, Fold};

use crate::models::bpe::MergeMap;
use crate::models::bpe::tables::At;

/// A bitmap of which codepoints fold, plus the symbols they fold to.
///
/// A codepoint fits in a u16, so there are 65536 of them and one bit each is 65536 bits = 1024
/// u64s = 8 KB. `rows` is that Vec of u64. Splitting a codepoint into a row and a column is just
/// dividing by 64 and taking the remainder, and 64 is a power of two, so it is a shift and a mask:
///
///   row = codepoint >> 6        col = codepoint & 0x3F
///
///   rows:  [    u64    |    u64    |    u64    | ... |    u64    ]   1024 rows, 8 KB
///            row 0       row 1       row 2             row 1023
///            cp 0..63     cp 64..127
///
///   one row is 64 codepoints, one bit each:
///
///   row 192:   bit 63 <-------------------------------------- bit 0
///              0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0
///                    ^                       ^
///                    col 60 folds            col 16 folds
///
/// A set bit means that codepoint folds to one symbol. `symbols` holds those symbols and nothing
/// else, packed in codepoint order: 13 KB instead of 256 KB for a flat `[u32; 65536]`. To find a
/// codepoint's slot in the `symbols` table, count the set bits before it: `row_start` has the count for all
/// earlier rows, and one popcount covers the bits before `col` in its own row.
pub struct SparseFold {
    /// One bit per codepoint: set iff it folds. If it does we emit the corresponding u32 directly.
    rows: Box<[u64]>,
    /// indexed by the row, indexes the symbols
    row_start: Box<[u32]>,
    /// The symbols of folding codepoints only, in codepoint order.
    symbols: Box<[u32]>,
    /// One-byte characters, which are always exactly one symbol whether they fold or not. 512 B.
    ascii: [u32; 128],
    /// The same mapping for codepoints past 0xFFFF (emoji, CJK ext). Too few and too spread out to
    /// be worth optimizing at all.
    non_bmp: AHashMap<char, u32>,
}

impl SparseFold {
    fn compact(
        codepoint_to_symbol: &[u32],
        byte_symbols: &[u32; 256],
        non_bmp: AHashMap<char, u32>,
    ) -> Self {
        let mut rows = vec![0u64; 1024];
        for (codepoint, &symbol) in codepoint_to_symbol.iter().enumerate() {
            if symbol != u32::MAX {
                // we set a single bit using |
                rows[codepoint >> 6] |= 1u64 << (codepoint & 0x3F);
            }
        }
        let mut row_start = vec![0u32; 1024];
        let mut seen = 0u32;
        for row in 0..1024 {
            row_start[row] = seen;
            seen += rows[row].count_ones();
        }
        let symbols: Vec<u32> = codepoint_to_symbol
            .iter()
            .copied()
            .filter(|&symbol| symbol != u32::MAX)
            .collect();
        let mut ascii = [0u32; 128];
        for (byte, symbol) in ascii.iter_mut().enumerate() {
            *symbol = if codepoint_to_symbol[byte] != u32::MAX {
                codepoint_to_symbol[byte]
            } else {
                byte_symbols[byte]
            };
        }
        Self {
            rows: rows.into_boxed_slice(),
            row_start: row_start.into_boxed_slice(),
            symbols: symbols.into_boxed_slice(),
            ascii,
            non_bmp,
        }
    }

    fn footprint(&self) -> usize {
        self.rows.len() * 8 + self.row_start.len() * 4 + self.symbols.len() * 4 + 512
    }

    /// The symbol at (row, col), or `u32::MAX` if that codepoint does not fold.
    #[inline(always)]
    fn get(&self, row: usize, col: u32) -> u32 {
        let bits = self.rows.at(row);
        if (bits >> col) & 1 == 0 {
            return u32::MAX;
        }
        let before =
            self.row_start.at(row) as usize + (bits & ((1u64 << col) - 1)).count_ones() as usize;
        self.symbols.at(before)
    }

    /// A one-byte character. Always one symbol, fold or not.
    #[inline(always)]
    pub fn get_ascii(&self, byte: u8) -> u32 {
        self.ascii.at((byte & 0x7F) as usize)
    }

    /// A character given as UTF-8 bytes. `u32::MAX` means it does not fold and the caller emits its
    /// bytes instead. `lead` and `char_len` come from the caller, which already has them.
    ///
    /// We use a small trick: any continuation byte can be converted to a key to index the sparse
    /// fold. For:
    ///   3 bytes:  1110xxxx 10yyyyyy 10zzzzzz     row = xxxx yyyyyy     col = zzzzzz
    ///             (0F)1111   111111(3F)
    ///
    ///   2 bytes:  110yyyyy 10zzzzzz              row =      yyyyy      col = zzzzzz
    ///            (1F)11111   111111(3F)
    #[inline(always)]
    pub fn get_bytes(&self, bytes: &[u8], start: usize, lead: u8, char_len: usize) -> u32 {
        match char_len {
            3 => self.get(
                (((lead & 0x0F) as usize) << 6) | (bytes.at(start + 1) & 0x3F) as usize,
                (bytes.at(start + 2) & 0x3F) as u32,
            ),
            2 => self.get((lead & 0x1F) as usize, (bytes.at(start + 1) & 0x3F) as u32),
            // four bytes: past the Basic Multilingual Plane, so the bitmap does not cover it
            _ => {
                let codepoint = (((lead & 0x07) as u32) << 18)
                    | (((bytes.at(start + 1) & 0x3F) as u32) << 12)
                    | (((bytes.at(start + 2) & 0x3F) as u32) << 6)
                    | (bytes.at(start + 3) & 0x3F) as u32;
                self.get_code(codepoint)
            }
        }
    }

    /// A character, for models whose atoms are characters rather than bytes.
    #[inline(always)]
    pub fn get_char(&self, character: char) -> u32 {
        self.get_code(character as u32)
    }

    #[inline(always)]
    fn get_code(&self, codepoint: u32) -> u32 {
        if codepoint < 0x10000 {
            // in that case we already have the codepoint so we need less masking than utf8.
            self.get(codepoint as usize >> 6, codepoint & 0x3F)
        } else {
            char::from_u32(codepoint)
                .and_then(|character| self.non_bmp.get(&character).copied())
                .unwrap_or(u32::MAX)
        }
    }
}

/// Builds the fold table, plus the `byte -> internal id` table the byte-level fallback path uses.
pub(super) fn build(
    vocab: &AHashMap<String, u32>,
    merges: &MergeMap,
    internal_id_map: &[u32],
    unmap: &[u32],
    byte_level: bool,
) -> (SparseFold, [u32; 256]) {
    // We don't create a hashmap for everything for memory efficiency.
    fn place(bmp: &mut [u32], non_bmp: &mut AHashMap<char, u32>, ch: char, id: u32) {
        if (ch as u32) < 0x10000 {
            bmp[ch as usize] = id;
        } else {
            non_bmp.insert(ch, id);
        }
    }

    // the flat 256 KB table is build-time only: `SparseFold::compact` shrinks it and it is dropped
    let mut cp_to_internal_id = vec![u32::MAX; 65536];
    let mut non_bmp: AHashMap<char, u32> = AHashMap::new();
    let (mut folded, mut unsafe_chars) = (0usize, 0usize);
    let mut byte_internal = [u32::MAX; 256];
    if byte_level {
        // A character reaches the merge loop as bytes, so folding it means proving the
        // merges are  predetermined. See `byte_level`.
        let folder = ByteLevelFold::new(vocab, merges, internal_id_map, unmap);
        byte_internal = folder.byte_internal();
        for (s, external) in vocab.iter() {
            match folder.fold(s, *external) {
                Fold::Folds(ch, id) => {
                    place(&mut cp_to_internal_id, &mut non_bmp, ch, id);
                    folded += 1;
                }
                Fold::Unsafe => unsafe_chars += 1,
                Fold::Skip => {}
            }
        }
    } else {
        // simple case, we just write the vocab tokens to a dense table instead of a HashMap.
        for (s, external) in vocab.iter() {
            let mut it = s.chars();
            if let (Some(ch), None) = (it.next(), it.next()) {
                let id = internal_id_map
                    .get(*external as usize)
                    .copied()
                    .unwrap_or(u32::MAX);
                place(&mut cp_to_internal_id, &mut non_bmp, ch, id);
                folded += 1;
            }
        }
    }
    info!("fold table: {folded} characters fold, {unsafe_chars} formable but boundary-unsafe");

    let fold = SparseFold::compact(&cp_to_internal_id, &byte_internal, non_bmp);
    info!(
        "fold table: {:.1} KB sparse (flat would be {:.1} KB)",
        fold.footprint() as f64 / 1024.0,
        65536.0 * 4.0 / 1024.0
    );
    (fold, byte_internal)
}
