//! Which characters a byte-level vocabulary can emit as one token instead of as their bytes.
//!
//! A byte-level model's atoms are the 256 bytes, so a multi-byte character like え reaches the
//! merge loop as three symbols that the merges then reassemble. When that reassembly is
//! predetermined, seeding the merge loop with the character's own token skips the work. Two
//! conditions make it predetermined:
//!
//! 1. The character's bytes must collapse into exactly one symbol when the merges are replayed
//!    the way BPE picks them: lowest rank first.
//! 2. No step of that replay may be taken over from outside the character. The merge loop does
//!    not know where the character ends: if a neighbouring symbol can merge with the character's
//!    first or last symbol at a lower rank, that merge fires first and the assembly never
//!    happens.
//!
//! A character failing either test gets no table entry; its bytes go through the merge loop as
//! usual, which is always exact. The fold is a shortcut, never a requirement.

use std::cmp;

use crate::models::bpe::MergeMap;
use crate::utils::byte_level::CHAR_BYTES_LOOKUP;

/// A character outside the table, or outside the Basic Multilingual Plane. The seeding loop
/// falls back to per-byte symbols when it reads this.
pub(super) const NO_FOLD: u32 = u32::MAX;

/// `table[codepoint]` is the id of the token this character folds to, or [`NO_FOLD`].
///
/// Indexed by Basic Multilingual Plane codepoints only: a `char` above `u16::MAX` never folds.
/// Single-byte characters are also left out, their byte symbol already is the seed.
pub(super) type FoldTable = Box<[u32; FOLD_TABLE_LEN]>;
pub(super) const FOLD_TABLE_LEN: usize = 1 << 16;

/// Builds the fold table for a byte-level vocabulary.
///
/// `vocab` still spells its tokens in byte-level characters ("Ã©" for é), which is what
/// [`ByteLevelFold::fold`] expects; call this before the store is rebuilt on raw bytes.
pub(super) fn build_fold_table(vocab: &[(String, u32)], merges: &MergeMap) -> FoldTable {
    let fold = ByteLevelFold::new(vocab, merges);
    let mut table = vec![NO_FOLD; FOLD_TABLE_LEN];
    for (token, id) in vocab {
        if let Fold::Folds(ch, id) = fold.fold(token, *id)
            && ch.len_utf8() > 1
            && (ch as usize) < FOLD_TABLE_LEN
        {
            table[ch as usize] = id;
        }
    }
    table
        .into_boxed_slice()
        .try_into()
        .expect("length is FOLD_TABLE_LEN")
}

/// What one vocab token is worth to the fold table.
pub(super) enum Fold {
    /// A single character whose bytes assemble to exactly this token, un-stealably.
    Folds(char, u32),
    /// Formable, but some step could be taken over by a neighbour.
    Unsafe,
    /// Not a single character, or its bytes never assemble at all. Nothing to record.
    Skip,
}

pub(super) struct ByteLevelFold<'a> {
    /// byte -> id of that byte's own one-character token. A byte's value is not its id
    /// (gpt2: 0x41 -> 32, 0x20 -> 220), which is why this indirection exists.
    byte_token: [u32; 256],
    /// `stolen_from_left[id]`: the lowest rank at which a left neighbour merges with `id`,
    /// counting only neighbours reachable at a character boundary. Same on the other side for
    /// `stolen_from_right`.
    stolen_from_left: Vec<u32>,
    stolen_from_right: Vec<u32>,
    merges: &'a MergeMap,
}

impl<'a> ByteLevelFold<'a> {
    pub(super) fn new(vocab: &[(String, u32)], merges: &'a MergeMap) -> Self {
        let mut byte_token = [u32::MAX; 256];
        for (token, id) in vocab {
            let mut chars = token.chars();
            if let (Some(ch), None) = (chars.next(), chars.next())
                && let Some(&b) = CHAR_BYTES_LOOKUP.get(&ch)
            {
                byte_token[b as usize] = *id;
            }
        }

        let (stolen_from_left, stolen_from_right) = boundary_merge_ranks(vocab, merges);

        Self {
            byte_token,
            stolen_from_left,
            stolen_from_right,
            merges,
        }
    }

    /// Verdict for `token`, whose id is `id`.
    pub(super) fn fold(&self, token: &str, id: u32) -> Fold {
        let Some(bytes) = token
            .chars()
            .map(|ch| CHAR_BYTES_LOOKUP.get(&ch).copied())
            .collect::<Option<Vec<u8>>>()
        else {
            // A character with no byte-level mapping: an added token, not text.
            return Fold::Skip;
        };
        // The table is keyed by codepoint, so only single-character tokens can go in it. This
        // also drops lone bytes >= 0x80, which are not characters on their own.
        let Ok(text) = std::str::from_utf8(&bytes) else {
            return Fold::Skip;
        };
        let mut it = text.chars();
        let (Some(ch), None) = (it.next(), it.next()) else {
            return Fold::Skip;
        };

        let mut running: Vec<u32> = bytes.iter().map(|&b| self.byte_token[b as usize]).collect();
        if running.contains(&u32::MAX) {
            return Fold::Skip; // a byte with no token of its own: never assemblable
        }
        // Replay the assembly the way the merge loop picks: the lowest rank among the pairs
        // still standing, until one symbol is left.
        while running.len() > 1 {
            let mut best: Option<(usize, u32, u32)> = None;
            for i in 0..running.len() - 1 {
                let pair = (running[i], running[i + 1]);
                if let Some((rank, product)) = self.merges.get(&pair)
                    && best.is_none_or(|(_, best_rank, _)| *rank < best_rank)
                {
                    best = Some((i, *rank, *product));
                }
            }
            let Some((i, rank, product)) = best else {
                return Fold::Skip; // stuck above one symbol: reference BPE stops here too
            };
            if rank >= self.stolen_from_left[running[0] as usize]
                || rank >= self.stolen_from_right[*running.last().unwrap() as usize]
            {
                return Fold::Unsafe;
            }
            running[i] = product;
            running.remove(i + 1);
        }

        debug_assert_eq!(running[0], id);
        Fold::Folds(ch, running[0])
    }
}

/// A folded character's edges are character boundaries by construction, and UTF-8
/// pins down what may be there:
///
/// - right neighbour = the next character's FIRST byte -> ASCII or a lead byte, never 0x80..=0xBF
/// - left  neighbour = the previous character's LAST byte -> never a lead byte, so always < 0xC0
fn boundary_merge_ranks(vocab: &[(String, u32)], merges: &MergeMap) -> (Vec<u32>, Vec<u32>) {
    let max_id = vocab
        .iter()
        .map(|(_, id)| *id)
        .chain(merges.iter().flat_map(|((a, b), (_, id))| [*a, *b, *id]))
        .max()
        .map_or(0, |id| id as usize + 1);

    // First and last real byte of every token. 0xFF marks a token with no byte spelling (an
    // added token): it counts as a possible stealer on the right, and as none on the left,
    // which errs towards refusing a fold.
    let (mut first, mut last) = (vec![0xFFu8; max_id], vec![0xFFu8; max_id]);
    for (token, id) in vocab {
        let Some(bytes) = token
            .chars()
            .map(|c| CHAR_BYTES_LOOKUP.get(&c).copied())
            .collect::<Option<Vec<u8>>>()
        else {
            continue;
        };
        if let (Some(f), Some(l)) = (bytes.first(), bytes.last()) {
            first[*id as usize] = *f;
            last[*id as usize] = *l;
        }
    }
    // 0xC0/0xC1 are overlong lead bytes and cannot occur on either side.
    let starts_at_boundary = |id: u32| first[id as usize] < 0x80 || first[id as usize] >= 0xC2;
    let ends_at_boundary = |id: u32| last[id as usize] < 0xC0;

    let mut stolen_from_left = vec![u32::MAX; max_id];
    let mut stolen_from_right = vec![u32::MAX; max_id];
    for ((a, b), (rank, _)) in merges.iter() {
        if *a as usize >= max_id || *b as usize >= max_id {
            continue;
        }
        if starts_at_boundary(*b) {
            stolen_from_right[*a as usize] = cmp::min(stolen_from_right[*a as usize], *rank);
        }
        if ends_at_boundary(*a) {
            stolen_from_left[*b as usize] = cmp::min(stolen_from_left[*b as usize], *rank);
        }
    }
    (stolen_from_left, stolen_from_right)
}

#[cfg(test)]
mod test {
    use super::{ByteLevelFold, Fold, NO_FOLD, build_fold_table};
    use crate::models::bpe::MergeMap;
    use crate::utils::byte_level::BYTES_CHAR_LOOKUP;

    /// 'é' is U+00E9 = bytes C3 A9; both are printable latin-1, so the byte-level names are the
    /// identity chars 'Ã' and '©' and the vocab spells the character "Ã©".
    fn setup(extra_merge: bool) -> (Vec<(String, u32)>, MergeMap) {
        let vocab = Vec::from([
            ("Ã".to_string(), 0),  // byte 0xC3
            ("©".to_string(), 1),  // byte 0xA9
            ("Ã©".to_string(), 2), // the character é
            ("x".to_string(), 3),
            ("xÃ".to_string(), 4),
        ]);
        let mut merges = MergeMap::new();
        merges.insert((0, 1), (1, 2)); // Ã + © -> é at rank 1
        if extra_merge {
            // x + Ã at rank 0: a left neighbour "x" grabs our first byte first, so the
            // assembly of é never happens and folding it would be wrong.
            merges.insert((3, 0), (0, 4));
        }
        (vocab, merges)
    }

    #[test]
    fn folds_when_nothing_can_steal_an_edge() {
        let (vocab, merges) = setup(false);
        let f = ByteLevelFold::new(&vocab, &merges);
        assert!(matches!(f.fold("Ã©", 2), Fold::Folds('é', 2)));
    }

    #[test]
    fn rejects_a_boundary_steal() {
        let (vocab, merges) = setup(true);
        let f = ByteLevelFold::new(&vocab, &merges);
        assert!(matches!(f.fold("Ã©", 2), Fold::Unsafe));
    }

    #[test]
    fn skips_what_is_not_one_character() {
        let (vocab, merges) = setup(false);
        let f = ByteLevelFold::new(&vocab, &merges);
        assert!(matches!(f.fold("xÃ", 4), Fold::Skip)); // two characters once decoded
        assert!(matches!(f.fold("<|endoftext|>", 9), Fold::Skip)); // '<' is fine, '|' is not remapped
        assert!(matches!(f.fold("Ã", 0), Fold::Skip)); // lone 0xC3 is not valid UTF-8
        assert!(matches!(f.fold("x", 3), Fold::Folds('x', 3))); // ASCII needs no assembly
    }

    #[test]
    fn table_keeps_the_fold_and_drops_ascii() {
        let (vocab, merges) = setup(false);
        let table = build_fold_table(&vocab, &merges);
        assert_eq!(table['é' as usize], 2);
        // 'x' folds but single-byte characters stay out: their byte symbol already is the seed.
        assert_eq!(table['x' as usize], NO_FOLD);
    }

    // Byte-level merges in a gpt2-like encoding. Byte level rewrites the vocab so every byte is a
    // printable char; bytes 0x80..=0xA0 become U+0122.. and 0xAE..=0xFF stay themselves.
    // U+671D 朝 → E6 9C 9D -> 'æ','ľ','Ŀ'
    // U+65E5 日 → E6 97 A5 -> 'æ','Ĺ','¥'
    //
    // 朝 assembles with ('æ','ľ') and ('æľ','Ŀ'), so it may only merge if no merge pair can take
    // an edge symbol first. We add such a pair: ('Ŀ','æ') 9D E6, which appears in 朝朝 and 朝日
    // at the boundary `.. 9D | E6 ..`. It has to be a LEAD byte (E6) doing the stealing: the symbol
    // after a complete character is always the next character's first byte.
    enum Thief {
        None,
        Lead,
        Continuation,
    }

    fn cjk_vocab(thief: Thief) -> (Vec<(String, u32)>, MergeMap) {
        assert_eq!(
            [0xE6u8, 0x9C, 0x9D].map(|b| BYTES_CHAR_LOOKUP[b as usize]),
            ['æ', 'ľ', 'Ŀ']
        );
        let mut vocab = Vec::from([
            ("æ".to_string(), 0),   // E6
            ("ľ".to_string(), 1),   // 9C
            ("Ŀ".to_string(), 2),   // 9D
            ("æľ".to_string(), 3),  // E6 9C
            ("æľĿ".to_string(), 4), // E6 9C 9D = 朝
        ]);
        let mut merges = MergeMap::new();
        // (left, right) -> (rank, product). Ranks leave room below for the thief.
        merges.insert((0, 1), (1, 3)); // 'æ' + 'ľ'  -> "æľ"
        merges.insert((3, 2), (2, 4)); // "æľ" + 'Ŀ' -> 朝
        match thief {
            Thief::None => {}
            Thief::Lead => {
                vocab.push(("Ŀæ".to_string(), 5)); // 9D E6, straddles a character boundary
                merges.insert((2, 0), (0, 5)); // rank 0, below every step of 朝's assembly
            }
            Thief::Continuation => {
                vocab.push(("Ģ".to_string(), 5)); // 80
                vocab.push(("ĿĢ".to_string(), 6)); // 9D 80, never at a boundary
                merges.insert((2, 5), (0, 6));
            }
        }
        (vocab, merges)
    }

    #[test]
    fn folds_a_cjk_char_when_no_neighbour_can_steal() {
        let (vocab, merges) = cjk_vocab(Thief::None);
        let f = ByteLevelFold::new(&vocab, &merges);
        assert!(matches!(f.fold("æľĿ", 4), Fold::Folds('朝', 4)));
    }

    #[test]
    fn refuses_the_same_char_once_a_lead_byte_can_steal() {
        let (vocab, merges) = cjk_vocab(Thief::Lead);
        let f = ByteLevelFold::new(&vocab, &merges);
        assert!(matches!(f.fold("æľĿ", 4), Fold::Unsafe));
    }

    #[test]
    fn a_continuation_byte_cannot_steal_so_it_still_folds() {
        let (vocab, merges) = cjk_vocab(Thief::Continuation);
        let f = ByteLevelFold::new(&vocab, &merges);
        assert!(matches!(f.fold("æľĿ", 4), Fold::Folds('朝', 4)));
    }
}
