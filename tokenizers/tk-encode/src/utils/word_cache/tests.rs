use super::cache::*;
use super::ids::*;
use super::slot::*;
use super::tags::*;
use crate::utils::signature::Signature;

const SPILLED_BUDGET: usize = 1 << 16;

impl<'a> Lookup<'a> {
    pub fn hit(self) -> Option<&'a [u32]> {
        match self {
            Lookup::Hit(ids) => Some(ids.as_slice()),
            Lookup::Miss(_) => None,
        }
    }
}

/// How many words [`WordCache::emit_home_hits`] answers at the start of `words`.
fn answered(cache: &WordCache, words: &[&[u8]]) -> usize {
    let signatures: Vec<Signature> = words.iter().map(|word| Signature::new(word)).collect();
    let mut out = vec![0u32; MAX_RUN_IDS * words.len()];
    // SAFETY: `out` holds `MAX_RUN_IDS` ids per signature.
    let (answered, _) =
        unsafe { cache.emit_home_hits(&signatures, |i| words[i], out.as_mut_ptr()) };
    answered
}

fn store(cache: &mut WordCache, word: &[u8], ids: &[u32]) {
    if let Lookup::Miss(at) = cache.lookup(word) {
        cache.insert(at, word, ids.iter().copied());
    }
}

/// The first `n` probe words whose home slot falls in `range`, for tests that
/// need to pick where in the table their words land.
fn words_homed_in(cache: &WordCache, range: std::ops::Range<usize>, n: usize) -> Vec<Vec<u8>> {
    (0u32..)
        .map(|i| format!("w{i}").into_bytes())
        .filter(|word| range.contains(&cache.home(Signature::new(word))))
        .take(n)
        .collect()
}

/// Three ids per word, since fewer has its own test, and homes clear of the
/// table's end, which has its own test too.
#[test]
fn a_stored_word_is_found_again() {
    let mut cache = WordCache::new(1 << 8);
    let n_slots = cache.cached_words.len();
    let safe = WordCache::WINDOW_SIZE..n_slots - WordCache::WINDOW_SIZE;
    let words = words_homed_in(&cache, safe, 3);
    for (i, word) in words.iter().enumerate() {
        let base = 3 * i as u32;
        store(&mut cache, word, &[base, base + 1, base + 2]);
    }
    for (i, word) in words.iter().enumerate() {
        let base = 3 * i as u32;
        assert_eq!(
            cache.lookup(word).hit(),
            Some(&[base, base + 1, base + 2][..]),
            "{word:?}"
        );
    }
}

/// Home slots stop at `placement_mask`; the table holds [`WordCache::WINDOW_SIZE`]
/// extra slots so a window starting on the last home does not run out of entries.
/// Two words homed there force the second one into the extra slots; both have to
/// round trip anyway.
#[test]
fn words_homed_on_the_last_slot_round_trip() {
    let mut cache = WordCache::new(1 << 2);
    let last = cache.placement_mask as usize;
    let words = words_homed_in(&cache, last..last + 1, 2);
    for (i, word) in words.iter().enumerate() {
        store(&mut cache, word, &[i as u32; 3]);
    }
    for (i, word) in words.iter().enumerate() {
        assert_eq!(
            cache.lookup(word).hit(),
            Some(&[i as u32; 3][..]),
            "{word:?}"
        );
    }
}

/// Both sides of the [`Signature::BYTES`] boundary, crossed with both sides of the
/// three-id inline boundary. The words past sixteen bytes are the ones that store
/// bytes in the buffer; no shorter word exercises that path.
#[test]
fn every_key_and_slot_shape_round_trips() {
    let mut cache = WordCache::new(1 << 8);
    let cases: [(&[u8], Vec<u32>); 4] = [
        (b"short", vec![1, 2]),
        (b"short-spilled", (0..10).collect()),
        (b"a-word-past-sixteen-bytes", vec![3]),
        (b"another-word-past-sixteen-bytes", (10..30).collect()),
    ];
    for (word, ids) in &cases {
        store(&mut cache, word, ids);
    }
    for (word, ids) in &cases {
        assert_eq!(cache.lookup(word).hit(), Some(&ids[..]), "{word:?}");
    }
}

#[test]
fn a_very_long_word_round_trips() {
    let mut cache = WordCache::new(1 << 8);
    let word = vec![b'q'; 8192];
    let ids: Vec<u32> = (0..2000).collect();
    store(&mut cache, &word, &ids);
    assert_eq!(cache.lookup(&word).hit(), Some(&ids[..]));
}

/// A slot has 24 bits for the word's length. Two words past that limit would look alike
/// to a slot that truncated their lengths, so neither is cached.
#[test]
fn a_word_too_long_for_a_slot_is_never_cached() {
    let mut cache = WordCache::new(1 << 8);
    for len in [1 << 24, (1 << 24) + 5] {
        let word = vec![b'q'; len];
        store(&mut cache, &word, &[1, 2, 3, 4]);
        assert_eq!(cache.lookup(&word).hit(), None, "len {len}");
    }
}

/// [`CachedIds::write`] copies a fixed number of ids and reports the real one. Each shape
/// of hit: ids in the slot, ids in the buffer, the most ids one copy holds and one more. The
/// last word stored is the last record of the buffer, where the copy reads past its ids.
#[test]
fn a_hit_writes_the_ids_it_holds() {
    let mut cache = WordCache::new(1 << 8);
    let most: Vec<u32> = (100..100 + MAX_RUN_IDS as u32).collect();
    let one_more: Vec<u32> = (200..=200 + MAX_RUN_IDS as u32).collect();
    let cases: [(&[u8], &[u32]); 5] = [
        (b"inline", &[1, 2]),
        (b"most-ids", &most),
        (b"one-more-id", &one_more),
        (b"a-word-past-sixteen-bytes", &[3]),
        (b"last-stored", &[4, 5, 6, 7]),
    ];
    for (word, ids) in cases {
        store(&mut cache, word, ids);
    }
    for (word, ids) in cases {
        let Lookup::Hit(hit) = cache.lookup(word) else {
            panic!("{word:?} was stored");
        };
        let mut out = vec![u32::MAX; hit.room()];
        // SAFETY: `out` holds `hit.room()` ids.
        let written = unsafe { hit.write(out.as_mut_ptr()) };
        assert_eq!(&out[..written], ids, "{word:?}");
    }
}

/// [`WordCache::emit_home_hits`] is a shortcut for [`WordCache::lookup`]. It must write the
/// ids a lookup returns for each word it answers, whether they sit in the slot or in the
/// buffer, and stop at the first word it cannot answer: a word with more ids than
/// [`MAX_RUN_IDS`] ends the first run, a word that was never stored ends the second.
#[test]
fn a_run_of_home_hits_writes_what_lookup_returns() {
    let mut cache = WordCache::new(1 << 8);
    let too_many: Vec<u32> = (100..=100 + MAX_RUN_IDS as u32).collect();
    let stored: [(&[u8], &[u32]); 7] = [
        (b"the", &[1]),
        (b"sixteen-byte-wd?", &[2, 3, 4]),
        (b"a", &[5, 6]),
        (b"spilled", &[7, 8, 9, 10]),
        (b"a-word-past-sixteen-bytes", &[11, 12]),
        (b"too-many-ids", &too_many),
        (b"after", &[13]),
    ];
    for (word, ids) in stored {
        store(&mut cache, word, ids);
    }
    let words: Vec<&[u8]> = stored
        .iter()
        .map(|&(word, _)| word)
        .chain([&b"never-stored"[..]])
        .collect();
    let signatures: Vec<Signature> = words.iter().map(|word| Signature::new(word)).collect();
    let homes: std::collections::HashSet<usize> = signatures
        .iter()
        .map(|&signature| cache.home(signature))
        .collect();
    assert_eq!(
        homes.len(),
        words.len(),
        "pick words with distinct home slots"
    );

    let mut next = 0;
    let mut answers = Vec::new();
    while next < words.len() {
        let run = &signatures[next..];
        let mut out = vec![0u32; MAX_RUN_IDS * run.len()];
        // SAFETY: `out` holds `MAX_RUN_IDS` ids per signature of the run.
        let (answered, written) =
            unsafe { cache.emit_home_hits(run, |i| words[next + i], out.as_mut_ptr()) };
        let mut ids = out[..written].iter();
        for word in &words[next..next + answered] {
            let expected = cache
                .lookup(word)
                .hit()
                .expect("the run answered this word");
            let got: Vec<u32> = ids.by_ref().take(expected.len()).copied().collect();
            assert_eq!(got, expected, "{word:?}");
        }
        answers.push(answered);
        next += answered + 1;
    }
    assert_eq!(answers, [5, 1]);
}

/// A one-home table sends every word to slot 0. The run must not answer a word that has only
/// part of the stored word's signature: ten `a`s against nine (another length), against a
/// last byte changed (another tail), and against a first byte changed (another head). Once
/// with the ids in the slot, once with the ids in the buffer.
#[test]
fn a_run_does_not_answer_a_word_with_part_of_the_signature() {
    for ids in [&[1][..], &[1, 2, 3, 4]] {
        let mut cache = WordCache::new(1);
        store(&mut cache, b"aaaaaaaaaa", ids);
        assert_eq!(answered(&cache, &[b"aaaaaaaaaa"]), 1);
        for near in [&b"aaaaaaaaa"[..], b"aaaaaaaaab", b"baaaaaaaaa"] {
            assert_eq!(answered(&cache, &[near]), 0, "{near:?} with {ids:?}");
        }
    }
}

/// Two words past [`Signature::BYTES`] with one signature have one home slot. The run
/// answers the one stored there, and must not answer the other with its ids.
#[test]
fn a_run_does_not_answer_a_long_word_with_another_words_ids() {
    let mut cache = WordCache::new(1 << 8);
    let (first, second) = (b"01234567-first--89abcdef", b"01234567-second-89abcdef");
    assert_eq!(Signature::new(first), Signature::new(second));
    store(&mut cache, first, &[1]);
    assert_eq!(answered(&cache, &[first]), 1);
    assert_eq!(answered(&cache, &[second]), 0);
}

/// An evict leaves the word's slot behind with offsets into a buffer that no longer holds
/// its ids. The run must leave such a word to the lookup, which reports a miss.
#[test]
fn a_run_does_not_answer_a_word_whose_ids_were_evicted() {
    let mut cache = WordCache::new(1 << 6);
    store(&mut cache, b"a-spilled-word", &[1, 2, 3, 4]);
    assert_eq!(answered(&cache, &[b"a-spilled-word"]), 1);
    store(&mut cache, b"the-filler-word", &vec![9; SPILLED_BUDGET]);
    assert_eq!(answered(&cache, &[b"a-spilled-word"]), 0);
}

/// Two words past [`Signature::BYTES`] with the same length, start and end have the same
/// signature, home slot and tag. A lookup that stopped at the signature would hand one
/// word the other's ids. One id each, so the words are spilled only for their bytes.
///
/// The second word differs in one byte, each byte the signature leaves out in turn. The
/// lengths are the ones where the stored bytes change shape: see [`Uncovered`].
#[test]
fn words_with_one_signature_keep_their_own_ids() {
    for len in [17, 23, 24, 32, 33, 48, 49] {
        let first: Vec<u8> = (0..len as u8).collect();
        for at in 8..len - 8 {
            let mut second = first.clone();
            second[at] ^= 0x80;
            assert_eq!(Signature::new(&first), Signature::new(&second));

            let mut cache = WordCache::new(1 << 8);
            store(&mut cache, &first, &[1]);
            assert_eq!(cache.lookup(&second).hit(), None, "len {len} byte {at}");
            store(&mut cache, &second, &[2]);
            assert_eq!(cache.lookup(&first).hit(), Some(&[1][..]));
            assert_eq!(cache.lookup(&second).hit(), Some(&[2][..]));
        }
    }
}

/// One window shape per row: the needle in various lanes, an empty slot in
/// the middle, at the edges, or absent. Candidates past the first empty
/// lane must not be reported (no entry can live there, since inserts always
/// fill the first empty lane) and the empty lane itself is the placement.
#[test]
fn the_scan_reports_matches_before_the_first_empty_and_the_empty_itself() {
    let offset = 3;
    let cases: &[(&[u8], u16, Option<usize>)] = &[
        // needle at lanes 0 and 2, empty at 3: lane 5's match is out of reach
        (
            &[
                0xA7, 0x31, 0xA7, 0x00, 0x5F, 0xA7, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18,
                0x19, 0x1A,
            ],
            0b101,
            Some(3),
        ),
        // no empty slot: every match is reachable, no placement
        (
            &[
                0xA7, 0x31, 0xA7, 0x22, 0x5F, 0xA7, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18,
                0x19, 0xA7,
            ],
            0b1000000000100101,
            None,
        ),
        // empty in lane 0: nothing is reachable
        (
            &[
                0x00, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7,
                0xA7, 0xA7,
            ],
            0,
            Some(0),
        ),
        // every lane matches, empty in the last lane
        (
            &[
                0xA7, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7,
                0xA7, 0x00,
            ],
            0b0111_1111_1111_1111,
            Some(15),
        ),
    ];
    for (lanes, want_mask, want_empty) in cases {
        let mut table = vec![0xEEu8; offset];
        table.extend_from_slice(lanes);
        let window = Window::new(&table, offset);
        let (candidates, first_empty) = window.find_matches_and_first_empty(0xA7);
        let found: Vec<usize> = candidates.collect();
        let want: Vec<usize> = (0..16)
            .filter(|i| want_mask >> i & 1 == 1)
            .map(|i| i + offset)
            .collect();
        assert_eq!(found, want, "lanes {lanes:?}");
        assert_eq!(
            first_empty,
            want_empty.map(|i| i + offset),
            "lanes {lanes:?}"
        );
    }
}

/// The tag row marks a free slot with 0x00, so no live entry may carry that
/// tag: the scan stops looking at the first 0x00 it sees, and an insert
/// treats the slot as free space. Enough words that a tag built from a raw
/// hash byte would land on 0x00 hundreds of times.
#[test]
fn a_live_tag_is_never_the_empty_marker() {
    for i in 0..100_000u32 {
        let tag = tag_of(Signature::new(&i.to_le_bytes()));
        assert_ne!(tag, WordCache::EMPTY, "word {i}");
    }
}

/// A nonzero start, since every spill after the first has one and offsets that
/// only round trip from zero would still pass.
#[test]
fn a_spilled_words_offsets_round_trip() {
    let cached = WordCacheSlot::new_spilled(Signature::default(), (5, 9), 0);
    match cached.variant() {
        CacheSlotType::Spilled(word) => assert_eq!(word.ids_offsets(), (5, 9)),
        CacheSlotType::SelfContained(_) => {
            panic!("built as spilled, read back as self-contained")
        }
    }
}

#[test]
fn a_word_with_one_id_is_stored_self_contained() {
    let cached = WordCacheSlot::new_self_contained(Signature::default(), [42].into_iter());
    match cached.variant() {
        CacheSlotType::SelfContained(word) => assert_eq!(word.ids().as_slice(), &[42]),
        CacheSlotType::Spilled(_) => panic!("one id fits without spilling"),
    }
}

/// A tag is one byte of hash, so about one occupied slot in 256 carries the tag
/// of a word it does not hold. Too rare to hit by chance here, so the collision
/// is forged. The lookup must confirm the signature and keep walking, not trust
/// the tag.
#[test]
fn a_tag_collision_is_confirmed_against_the_signature() {
    let mut cache = WordCache::new(1 << 8);
    let signature = Signature::new(b"beta");
    let (index, tag) = (cache.home(signature), tag_of(signature));
    cache.quick_lookup[index] = tag;
    cache.cached_words[index] =
        WordCacheSlot::new_self_contained(Signature::new(b"decoy"), [7].into_iter());

    assert_eq!(cache.lookup(b"beta").hit(), None);
    store(&mut cache, b"beta", &[2]);
    assert_eq!(cache.lookup(b"beta").hit(), Some(&[2][..]));
}

/// A word whose whole window is taken is still cached: it evicts its home slot.
/// The other fifteen words in the window keep their ids.
#[test]
fn a_full_window_evicts_only_the_home_slot() {
    // A one-home table: placement_mask is 0, so every word homes at slot 0 and
    // the sixteen slots from there are one shared window.
    let mut cache = WordCache::new(1);
    let words = words_homed_in(&cache, 0..1, WordCache::WINDOW_SIZE);
    for (i, word) in words.iter().enumerate() {
        store(&mut cache, word, &[i as u32]);
    }

    store(&mut cache, b"newcomer", &[999]);
    assert_eq!(cache.lookup(b"newcomer").hit(), Some(&[999][..]));
    // Inserts fill the window in order, so the home slot holds the first word.
    assert_eq!(cache.lookup(&words[0]).hit(), None);
    for (i, word) in words.iter().enumerate().skip(1) {
        assert_eq!(cache.lookup(word).hit(), Some(&[i as u32][..]), "{word:?}");
    }
}

/// The filler word brings a whole budget of ids on its own, so caching it
/// evicts whatever the buffer holds. The evicted word's slot keeps its tag
/// and signature, but its ids are gone from the buffer: looking the word up
/// again has to be a miss, not a hit on the filler's ids.
#[test]
fn a_spilled_word_misses_after_an_evict() {
    let mut cache = WordCache::new(1 << 6);
    store(&mut cache, b"a-spilled-word", &[1, 2, 3, 4]);
    store(&mut cache, b"the-filler-word", &vec![9; SPILLED_BUDGET]);
    assert_eq!(cache.lookup(b"a-spilled-word").hit(), None);
}

/// An evict drops a word's ids but leaves its slot behind. The miss the
/// word's lookup then returns must be a placement `insert` accepts, and
/// the word must round trip through it. New ids for the second insertion,
/// so a hit on the first ones cannot pass.
#[test]
fn an_evicted_word_round_trips_once_re_inserted() {
    let mut cache = WordCache::new(1 << 6);
    store(&mut cache, b"a-spilled-word", &[1, 2, 3, 4]);
    store(&mut cache, b"the-filler-word", &vec![9; SPILLED_BUDGET]);
    store(&mut cache, b"a-spilled-word", &[5, 6, 7, 8]);
    assert_eq!(
        cache.lookup(b"a-spilled-word").hit(),
        Some(&[5, 6, 7, 8][..])
    );
}

/// An evict leaves the word's slot behind, signature and tag intact. The miss the
/// word's lookup then returns must place the fresh ids back into that slot,
/// not into an empty one: the window stays clean of stale copies, which is
/// what lets the walk stop at the first match.
#[test]
fn a_miss_reuses_the_slot_of_its_stale_copy() {
    let mut cache = WordCache::new(1 << 6);
    let slot = match cache.lookup(b"a-spilled-word") {
        Lookup::Miss(at) => at.index,
        Lookup::Hit(_) => panic!("the cache is empty, this must be a miss"),
    };
    store(&mut cache, b"a-spilled-word", &[1, 2, 3, 4]);
    store(&mut cache, b"the-filler-word", &vec![9; SPILLED_BUDGET]);
    match cache.lookup(b"a-spilled-word") {
        Lookup::Miss(at) => assert_eq!(at.index, slot),
        Lookup::Hit(_) => panic!("the ids were evicted, this must be a miss"),
    }
}

/// Five evict cycles on the same word: it must end up holding exactly one
/// slot. A placement that preferred an empty slot would leave one more
/// stale copy behind per cycle.
#[test]
fn an_evicted_word_never_occupies_two_slots() {
    let mut cache = WordCache::new(1 << 6);
    for round in 0..5u32 {
        store(&mut cache, b"a-spilled-word", &[round; 4]);
        store(&mut cache, b"the-filler-word", &vec![9; SPILLED_BUDGET]);
    }
    let signature = Signature::new(b"a-spilled-word");
    let copies = cache
        .cached_words
        .iter()
        .filter(|slot| slot.has_signature(signature))
        .count();
    assert_eq!(copies, 1);
}

/// A self-contained word keeps its ids in the slot, not in the buffer, so
/// an evict must not cost it its hit. Two filler words, because the first
/// fills an empty buffer exactly to the budget; the second overflows it.
#[test]
fn a_self_contained_word_still_hits_after_an_evict() {
    let mut cache = WordCache::new(1 << 6);
    store(&mut cache, b"short", &[1, 2]);
    store(&mut cache, b"the-filler-word", &vec![9; SPILLED_BUDGET]);
    store(&mut cache, b"one-more-word", &[3, 4, 5, 6]);
    assert_eq!(cache.lookup(b"short").hit(), Some(&[1, 2][..]));
}

/// The evict happens inside the insert that caches this word: the slot
/// must carry the generation the evict moved to, not the one the insert
/// started with, or the word would be stale the moment it is cached.
#[test]
fn the_evicting_insert_caches_its_own_word() {
    let mut cache = WordCache::new(1 << 6);
    store(&mut cache, b"the-filler-word", &vec![9; SPILLED_BUDGET]);
    store(&mut cache, b"a-spilled-word", &[1, 2, 3, 4]);
    assert_eq!(
        cache.lookup(b"a-spilled-word").hit(),
        Some(&[1, 2, 3, 4][..])
    );
}

/// The generation counter wraps back to zero after 2^32 evicts, where a
/// slot stamped long ago would look fresh again and hit on another word's
/// ids. The wrap therefore clears the whole table, self-contained slots
/// included; only the word the wrapping insert caches survives it.
#[test]
fn the_evict_that_wraps_the_generation_clears_the_table() {
    let mut cache = WordCache::new(1 << 6);
    cache.spilled_generation = u32::MAX;
    store(&mut cache, b"short", &[1, 2]);
    store(&mut cache, b"a-spilled-word", &[1, 2, 3, 4]);
    store(&mut cache, b"the-filler-word", &vec![9; SPILLED_BUDGET]);
    assert_eq!(cache.lookup(b"short").hit(), None);
    assert_eq!(cache.lookup(b"a-spilled-word").hit(), None);
    assert_eq!(
        cache.lookup(b"the-filler-word").hit(),
        Some(&vec![9; SPILLED_BUDGET][..])
    );
}

/// Placements are built by hand so nothing but `insert` is on trial.
#[test]
fn the_spilled_buffer_never_outgrows_its_budget() {
    let mut cache = WordCache::new(1 << 6);
    let word = b"a-word-past-sixteen-bytes";
    let signature = Signature::new(word);
    let tag = tag_of(signature);
    for i in 0..3 * (SPILLED_BUDGET / 8) {
        let at = InsertPlacement {
            index: 0,
            signature,
            tag,
        };
        cache.insert(at, word, [i as u32; 8].into_iter());
    }
    assert!(
        cache.spilled_buffer.len() <= SPILLED_BUDGET,
        "the buffer reached {} of a {} budget",
        cache.spilled_buffer.len(),
        SPILLED_BUDGET
    );
}

/// Churn a table far too small for its input, mixing every key and slot shape,
/// and check the one hard promise: the cache may forget a word, it must never
/// answer with another word's ids.
#[test]
fn a_hit_never_returns_another_words_ids() {
    let mut cache = WordCache::new(64);
    let mut expected: Vec<(Vec<u8>, Vec<u32>)> = Vec::new();
    for i in 0..2000usize {
        let word = match i % 4 {
            0 => format!("w{i}"),
            1 => format!("a-long-word-past-sixteen-bytes-{i}"),
            2 => format!("k{i}xxxxxxxxxxxx"),
            _ => format!("{}-{i}", "z".repeat(i % 40)),
        }
        .into_bytes();
        // No two words share an id (k stays below 16), so a wrong hit cannot
        // return the right ids by luck.
        let ids: Vec<u32> = (0..=(i % 9) as u32).map(|k| i as u32 * 16 + k).collect();
        store(&mut cache, &word, &ids);
        expected.push((word, ids));
    }

    let mut live = 0;
    for (word, ids) in &expected {
        if let Some(hit) = cache.lookup(word).hit() {
            assert_eq!(hit, &ids[..], "{word:?}");
            live += 1;
        }
    }
    assert!(live > 0, "everything was evicted, the test proves nothing");
}
