//! Which characters a byte-level vocab can emit as one token instead of as their individual bytes.
//!
//! A byte-level model's atoms are the 256 bytes, so the character え reaches the merge loop as
//! three symbols that then merge back together.
//! If the assembly is *predetermined* we can skip it: seed the merge loop with the character's
//! token directly.
//!
//! 1. The bytes must collapse to exactly one symbol, replayed the way BPE picks --
//!    lowest rank, leftmost on a tie.
//! 2. No step may be pre-emptable by a token *outside* the character. Bytes do not know where
//!    the character ends: if a left neighbour can merge with our first symbol at a lower rank,
//!    it fires first and the assembly never happens.
//! Fail either test and the character simply gets no entry: the encoder emits its bytes and the
//! merge loop assembles them, which is always exact. The fold is a shortcut, never a

use ahash::AHashMap;
use std::cmp;

use crate::models::bpe::MergeMap;
use crate::utils::byte_level::{BYTES_CHAR_LOOKUP, CHAR_BYTES_LOOKUP};

/// What one vocab token is worth to the fold table.
pub(super) enum Fold {
    /// A single character whose bytes assemble to exactly this token, un-stealably.
    Folds(char, u32),
    /// Formable, but some step could be pre-empted by a neighbour. Worth counting: a high count
    /// means the vocab has lots of near-misses, not that the fold is broken.
    Unsafe,
    /// Not a single character, or its bytes never assemble at all. Nothing to record.
    Skip,
}

pub(super) struct ByteLevelFold<'a> {
    /// byte -> internal id of that byte's own one-character token. A byte's VALUE is not its
    /// external id (gpt2: 0x41 -> 32, 0x20 -> 220), which is why this indirection exists.
    byte_internal: [u32; 256],
    /// Lowest rank at which the symbol can be taken from the left / right, counting only
    /// neighbours reachable at a character boundary.
    merge_rank_left: Vec<u32>,
    merge_rank_right: Vec<u32>,
    merges: &'a MergeMap,
    internal_id_map: &'a [u32],
    unmap: &'a [u32],
}

impl<'a> ByteLevelFold<'a> {
    /// byte -> internal id of that byte's own token. Needed by the encoder for the fallback path:
    /// a character that does not fold is emitted as its bytes.
    pub(super) fn byte_internal(&self) -> [u32; 256] {
        self.byte_internal
    }

    pub(super) fn new(
        vocab: &AHashMap<String, u32>,
        merges: &'a MergeMap,
        internal_id_map: &'a [u32],
        unmap: &'a [u32],
    ) -> Self {
        let iid = |external: u32| {
            internal_id_map
                .get(external as usize)
                .copied()
                .unwrap_or(u32::MAX)
        };

        let mut byte_internal = [u32::MAX; 256];
        let mut buf = [0u8; 4];
        for b in 0..256usize {
            if let Some(&external) = vocab.get(&*BYTES_CHAR_LOOKUP[b].encode_utf8(&mut buf)) {
                byte_internal[b] = iid(external);
            }
        }

        let (merge_rank_left, merge_rank_right) =
            boundary_merge_ranks(vocab, merges, internal_id_map, unmap.len());

        Self {
            byte_internal,
            merge_rank_left,
            merge_rank_right,
            merges,
            internal_id_map,
            unmap,
        }
    }

    fn iid(&self, external: u32) -> u32 {
        self.internal_id_map
            .get(external as usize)
            .copied()
            .unwrap_or(u32::MAX)
    }

    /// Verdict for `token`, whose external id is `external`.
    pub(super) fn fold(&self, token: &str, external: u32) -> Fold {
        let Some(bytes) = token
            .chars()
            .map(|ch| CHAR_BYTES_LOOKUP.get(&ch).copied())
            .collect::<Option<Vec<u8>>>()
        else {
            // if any of the byte was not in the lookup return
            return Fold::Skip;
        };
        // The table is keyed by codepoint, so only single-character tokens can go in it. Note
        // this also drops lone bytes >= 0x80, which are not characters on their own.
        let Ok(text) = std::str::from_utf8(&bytes) else {
            return Fold::Skip;
        };
        let mut it = text.chars();
        let (Some(ch), None) = (it.next(), it.next()) else {
            return Fold::Skip;
        };

        let mut running: Vec<u32> = bytes
            .iter()
            .map(|&b| self.byte_internal[b as usize])
            .collect();
        if running.contains(&u32::MAX) {
            return Fold::Skip; // a byte with no token of its own: never assemblable
        }
        // Now we loop over the bytes of the char and apply bpe: loop on global merge, merge then
        // loop on global merge, merge, etc
        while running.len() > 1 {
            let mut best: Option<(usize, u32, u32)> = None;
            // we loop on the ranks of the different global merges and comput the best
            for i in 0..running.len() - 1 {
                let pair = (
                    self.unmap[running[i] as usize],
                    self.unmap[running[i + 1] as usize],
                );
                if let Some((rank, product)) = self.merges.get(&pair)
                    && best.is_none_or(|(_, best_rank, _)| *rank < best_rank)
                {
                    best = Some((i, *rank, self.iid(*product)));
                }
            }
            let Some((i, rank, product)) = best else {
                return Fold::Skip; // stuck above one symbol: reference BPE stops here too
            };
            if rank >= self.merge_rank_right[running[0] as usize]
                || rank >= self.merge_rank_left[*running.last().unwrap() as usize]
            {
                return Fold::Unsafe;
            }
            running[i] = product;
            running.remove(i + 1);
        }

        debug_assert_eq!(running[0], self.iid(external));
        Fold::Folds(ch, running[0])
    }
}

/// A folded character's edges are character boundaries by construction, and UTF-8
/// pins down what may be there:
///
/// - right neighbour = the next character's FIRST byte -> ASCII or a lead byte, never 0x80..=0xBF
/// - left  neighbour = the previous character's LAST byte -> never a lead byte, so always < 0xC0
fn boundary_merge_ranks(
    vocab: &AHashMap<String, u32>,
    merges: &MergeMap,
    internal_id_map: &[u32],
    n_internal: usize,
) -> (Vec<u32>, Vec<u32>) {
    // First and last real byte of every token.
    let (mut first, mut last) = (vec![0xFFu8; n_internal], vec![0xFFu8; n_internal]);
    for (token, external) in vocab {
        let Some(i) = internal_id_map
            .get(*external as usize)
            .copied()
            .filter(|i| (*i as usize) < n_internal)
        else {
            continue;
        };
        let Some(bytes) = token
            .chars()
            .map(|c| CHAR_BYTES_LOOKUP.get(&c).copied())
            .collect::<Option<Vec<u8>>>()
        else {
            continue;
        };
        if let (Some(f), Some(l)) = (bytes.first(), bytes.last()) {
            first[i as usize] = *f;
            last[i as usize] = *l;
        }
    }
    // 0xC0/0xC1 are overlong lead bytes and cannot occur on either side.
    let starts_at_boundary = |i: u32| first[i as usize] < 0x80 || first[i as usize] >= 0xC2;
    let ends_at_boundary = |i: u32| last[i as usize] < 0xC0;

    let iid = |external: u32| {
        internal_id_map
            .get(external as usize)
            .copied()
            .unwrap_or(u32::MAX)
    };
    let mut left = vec![u32::MAX; internal_id_map.len()];
    let mut right = vec![u32::MAX; internal_id_map.len()];
    for ((a, b), (rank, _)) in merges.iter() {
        let (ia, ib) = (iid(*a), iid(*b));
        if ia == u32::MAX
            || ib == u32::MAX
            || ia as usize >= n_internal
            || ib as usize >= n_internal
        {
            continue;
        }
        if starts_at_boundary(ib) {
            left[ia as usize] = cmp::min(left[ia as usize], *rank);
        }
        if ends_at_boundary(ia) {
            right[ib as usize] = cmp::min(right[ib as usize], *rank);
        }
    }
    (left, right)
}

#[cfg(test)]
mod test {
    use super::{ByteLevelFold, Fold};
    use crate::models::bpe::MergeMap;
    use crate::utils::byte_level::BYTES_CHAR_LOOKUP;
    use ahash::AHashMap;

    /// 'é' is U+00E9 = bytes C3 A9; both are printable latin-1, so the byte-level names are the
    /// identity chars 'Ã' and '©' and the vocab spells the character "Ã©".
    fn setup(extra_merge: bool) -> (AHashMap<String, u32>, MergeMap) {
        let vocab = AHashMap::from_iter(vec![
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
        let ids = [0, 1, 2, 3, 4];
        let f = ByteLevelFold::new(&vocab, &merges, &ids, &ids);
        assert!(matches!(f.fold("Ã©", 2), Fold::Folds('é', 2)));
    }

    #[test]
    fn rejects_a_boundary_steal() {
        let (vocab, merges) = setup(true);
        let ids = [0, 1, 2, 3, 4];
        let f = ByteLevelFold::new(&vocab, &merges, &ids, &ids);
        assert!(matches!(f.fold("Ã©", 2), Fold::Unsafe));
    }

    #[test]
    fn skips_what_is_not_one_character() {
        let (vocab, merges) = setup(false);
        let ids = [0, 1, 2, 3, 4];
        let f = ByteLevelFold::new(&vocab, &merges, &ids, &ids);
        assert!(matches!(f.fold("xÃ", 4), Fold::Skip)); // two characters once decoded
        assert!(matches!(f.fold("<|endoftext|>", 9), Fold::Skip)); // '<' is fine, '|' is not remapped
        assert!(matches!(f.fold("Ã", 0), Fold::Skip)); // lone 0xC3 is not valid UTF-8
        assert!(matches!(f.fold("x", 3), Fold::Folds('x', 3))); // ASCII needs no assembly
    }

    // Byte-level merges in a gpt2-like encoding. Byte level rewrites the vocab so every byte is a
    // printable char; bytes 0x80..=0xA0 become U+0122.. and 0xAE..=0xFF stay themselves.
    // U+671D 朝 → E6 9C 9D -> 'æ','ľ','Ŀ'
    // U+65E5 日 → E6 97 A5 -> 'æ','Ĺ','¥'
    //
    // 朝 assembles with ('æ','ľ') and ('æľ','Ŀ'), so it may only merge if no merge pair can take
    // an edge symbol first. We add such a pair:('Ŀ','æ') 9D E6, which appear in 朝朝 and 朝日
    // at the boundary `.. 9D | E6 ..`. It has to be a LEAD byte (E6) doing the stealing: the symbol
    // after a complete character is always the next character's first byte.
    enum Thief {
        None,
        Lead,
        Continuation,
    }

    fn cjk_vocab(thief: Thief) -> (AHashMap<String, u32>, MergeMap, Vec<u32>) {
        assert_eq!(
            [0xE6u8, 0x9C, 0x9D].map(|b| BYTES_CHAR_LOOKUP[b as usize]),
            ['æ', 'ľ', 'Ŀ']
        );
        let mut vocab = AHashMap::from_iter(vec![
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
                vocab.insert("Ŀæ".to_string(), 5); // 9D E6, straddles a character boundary
                merges.insert((2, 0), (0, 5)); // rank 0, below every step of 朝's assembly
            }
            Thief::Continuation => {
                vocab.insert("Ģ".to_string(), 5); // 80
                vocab.insert("ĿĢ".to_string(), 6); // 9D 80, never at a boundary
                merges.insert((2, 5), (0, 6));
            }
        }
        let ids = (0..vocab.len() as u32).collect();
        (vocab, merges, ids)
    }

    #[test]
    fn folds_a_cjk_char_when_no_neighbour_can_steal() {
        let (vocab, merges, ids) = cjk_vocab(Thief::None);
        let f = ByteLevelFold::new(&vocab, &merges, &ids, &ids);
        assert!(matches!(f.fold("æľĿ", 4), Fold::Folds('朝', 4)));
    }

    #[test]
    fn refuses_the_same_char_once_a_lead_byte_can_steal() {
        let (vocab, merges, ids) = cjk_vocab(Thief::Lead);
        let f = ByteLevelFold::new(&vocab, &merges, &ids, &ids);
        assert!(matches!(f.fold("æľĿ", 4), Fold::Unsafe));
    }

    #[test]
    fn a_continuation_byte_cannot_steal_so_it_still_folds() {
        let (vocab, merges, ids) = cjk_vocab(Thief::Continuation);
        let f = ByteLevelFold::new(&vocab, &merges, &ids, &ids);
        assert!(matches!(f.fold("æľĿ", 4), Fold::Folds('朝', 4)));
    }
}
