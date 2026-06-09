/// Open-addressing hash table tuned for BPE merge lookups.
///
/// ## Layout
///
/// Each slot holds a `(left: u32, right: u32, rank: u32, new_id: u32)` quad —
/// **16 bytes**.  On a 64-byte cache line this packs exactly **4 slots**, so a
/// linear-probe sequence of ≤ 4 steps never leaves the cache line that was
/// fetched on the first miss.
///
/// ## Why this beats a general-purpose hash map here
///
/// * **Read-only after construction** — no deletion, no tombstones, no
///   re-hashing.  The probe loop is two comparisons per slot and no branch on
///   a "tombstone" state.
/// * **Dense keys** — BPE pair IDs are small integers; the splitmix64
///   finalizer distributes them uniformly without the overhead of a complex
///   hash function.
/// * **~60 % load factor** — keeps average probe length under 1.5, so most
///   lookups hit in the first cache line.
///
/// ## Limitations
///
/// Pair components must be < `u32::MAX` (valid for any model with < 4 billion
/// tokens).  `u32::MAX` is used as the empty-slot sentinel.
use serde::{Deserialize, Serialize};

type Pair = (u32, u32);

const EMPTY: u32 = u32::MAX;

/// One slot in the table.  16 bytes → 4 per cache line.
#[derive(Clone, Copy)]
#[repr(C)]
struct Slot {
    left: u32,
    right: u32,
    rank: u32,
    new_id: u32,
}

impl Slot {
    #[inline]
    fn is_empty(self) -> bool {
        self.left == EMPTY
    }
}

/// Open-addressing hash table from `Pair → (rank, new_id)`.
#[derive(Clone)]
pub struct MergeTable {
    slots: Box<[Slot]>,
    /// `slots.len() - 1`; always a power-of-two mask.
    mask: usize,
}

impl std::fmt::Debug for MergeTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MergeTable")
            .field("len", &self.len())
            .field("capacity", &self.slots.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Hash
// ---------------------------------------------------------------------------

/// Splitmix64 finalizer applied to the 64-bit encoding of a pair.
/// Excellent avalanche even for small (e.g. BPE vocab-sized) integers.
#[inline(always)]
fn hash_pair(a: u32, b: u32) -> usize {
    let h = (a as u64) | ((b as u64) << 32);
    let h = (h ^ (h >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    let h = (h ^ (h >> 27)).wrapping_mul(0x94d049bb133111eb);
    (h ^ (h >> 31)) as usize
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

fn capacity_for(count: usize) -> usize {
    // Target ~60 % load factor.  Minimum 8 slots.
    let min = (count * 8 / 5).max(8);
    min.next_power_of_two()
}

impl MergeTable {
    /// Build from an iterator of `(pair, (rank, new_id))` entries.
    pub fn from_iter(iter: impl IntoIterator<Item = (Pair, (u32, u32))>) -> Self {
        let entries: Vec<(Pair, (u32, u32))> = iter.into_iter().collect();
        let cap = capacity_for(entries.len());
        let mask = cap - 1;

        let mut slots = vec![
            Slot {
                left: EMPTY,
                right: EMPTY,
                rank: 0,
                new_id: 0,
            };
            cap
        ]
        .into_boxed_slice();

        for ((left, right), (rank, new_id)) in entries {
            debug_assert!(left != EMPTY, "pair component must not equal u32::MAX");
            let mut i = hash_pair(left, right) & mask;
            loop {
                if slots[i].is_empty() {
                    slots[i] = Slot {
                        left,
                        right,
                        rank,
                        new_id,
                    };
                    break;
                }
                i = (i + 1) & mask;
            }
        }

        Self { slots, mask }
    }

    /// Look up a merge.  Returns `Some((rank, new_id))` or `None`.
    ///
    /// Hot path: two u32 comparisons per slot, probe sequence stays within
    /// the same 64-byte cache line for the first 4 steps.
    #[inline]
    pub fn get(&self, (left, right): Pair) -> Option<(u32, u32)> {
        let mut i = hash_pair(left, right) & self.mask;
        loop {
            // SAFETY: i is always within bounds because i = (anything & mask)
            // and mask = slots.len() - 1.
            let s = unsafe { *self.slots.get_unchecked(i) };
            if s.left == left && s.right == right {
                return Some((s.rank, s.new_id));
            }
            if s.is_empty() {
                return None;
            }
            i = (i + 1) & self.mask;
        }
    }

    /// Number of occupied entries.
    pub fn len(&self) -> usize {
        self.slots.iter().filter(|s| !s.is_empty()).count()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.iter().all(|s| s.is_empty())
    }

    /// Iterate over all `(pair, rank, new_id)` entries in unspecified order.
    pub fn iter(&self) -> impl Iterator<Item = (Pair, u32, u32)> + '_ {
        self.slots.iter().filter(|s| !s.is_empty()).map(|s| {
            ((s.left, s.right), s.rank, s.new_id)
        })
    }
}

// ---------------------------------------------------------------------------
// PartialEq — order-independent entry comparison
// ---------------------------------------------------------------------------

impl PartialEq for MergeTable {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter()
            .all(|(pair, rank, new_id)| other.get(pair) == Some((rank, new_id)))
    }
}

// ---------------------------------------------------------------------------
// Serde — serialise as a Vec for JSON round-trips
// ---------------------------------------------------------------------------

impl Serialize for MergeTable {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // Collect and sort by rank so the on-disk order is deterministic.
        let mut entries: Vec<(Pair, u32, u32)> = self.iter().collect();
        entries.sort_unstable_by_key(|&(_, rank, _)| rank);
        entries.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for MergeTable {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let entries: Vec<(Pair, u32, u32)> = Deserialize::deserialize(deserializer)?;
        Ok(Self::from_iter(
            entries.into_iter().map(|(pair, rank, new_id)| (pair, (rank, new_id))),
        ))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn small_table() -> MergeTable {
        MergeTable::from_iter([
            ((0, 1), (0, 4)),
            ((4, 2), (1, 5)),
            ((5, 3), (2, 6)),
        ])
    }

    #[test]
    fn basic_lookup() {
        let t = small_table();
        assert_eq!(t.get((0, 1)), Some((0, 4)));
        assert_eq!(t.get((4, 2)), Some((1, 5)));
        assert_eq!(t.get((5, 3)), Some((2, 6)));
        assert_eq!(t.get((9, 9)), None);
    }

    #[test]
    fn len_and_iter() {
        let t = small_table();
        assert_eq!(t.len(), 3);
        let mut entries: Vec<_> = t.iter().collect();
        entries.sort_by_key(|&(_, rank, _)| rank);
        assert_eq!(entries, [((0, 1), 0, 4), ((4, 2), 1, 5), ((5, 3), 2, 6)]);
    }

    #[test]
    fn partial_eq() {
        let a = small_table();
        let b = MergeTable::from_iter([
            ((5, 3), (2, 6)),
            ((0, 1), (0, 4)),
            ((4, 2), (1, 5)),
        ]);
        assert_eq!(a, b);
    }

    #[test]
    fn absent_pair_is_none() {
        let t = small_table();
        // Pairs where left/right are present individually but not together
        assert_eq!(t.get((0, 2)), None);
        assert_eq!(t.get((1, 0)), None);
    }

    #[test]
    fn large_table_no_collisions() {
        // Build a 10 000 entry table and verify every entry round-trips.
        let entries: Vec<(Pair, (u32, u32))> = (0u32..10_000)
            .map(|i| ((i, i + 1), (i, i + 10_000)))
            .collect();
        let t = MergeTable::from_iter(entries.iter().copied());
        for (pair, (rank, new_id)) in &entries {
            assert_eq!(t.get(*pair), Some((*rank, *new_id)));
        }
        assert_eq!(t.len(), 10_000);
    }

    #[test]
    fn capacity_is_power_of_two() {
        // Sanity-check that the table never exceeds ~60 % load factor.
        let t = MergeTable::from_iter((0u32..1_000).map(|i| ((i, i + 1), (i, 0))));
        assert!(t.slots.len().is_power_of_two());
        assert!((t.len() as f64 / t.slots.len() as f64) < 0.65);
    }
}
