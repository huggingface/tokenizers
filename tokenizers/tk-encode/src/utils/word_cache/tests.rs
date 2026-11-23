use super::ids::MAX_RUN_IDS;
use super::run::HomeSlot;
use super::table::{Lookup, WordCache};
use super::tags::{WINDOW_SIZE, tag_of};
use crate::pipeline::Token;
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

/// How many words [`WordCache::emit_run`] answers at the start of `words`.
fn answered(cache: &WordCache, words: &[&[u8]]) -> usize {
    let signatures: Vec<Signature> = words.iter().map(|word| Signature::new(word)).collect();
    cache.emit_run(&signatures, |i| words[i], &mut Vec::new()).0
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
    let n_slots = cache.slots.len();
    let safe = WINDOW_SIZE..n_slots - WINDOW_SIZE;
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

/// The table holds [`WINDOW_SIZE`] slots past its last home slot, so a window starting
/// there does not run out of entries.
/// Two words homed there force the second one into the extra slots; both have to
/// round trip anyway.
#[test]
fn words_homed_on_the_last_slot_round_trip() {
    let mut cache = WordCache::new(1 << 2);
    let last = cache.slots.len() - WINDOW_SIZE - 1;
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
    // The run reads the slot after the last home slot for the second word.
    assert_eq!(answered(&cache, &[&words[1], &words[0]]), 2);
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

/// [`CachedIds::append_to`] copies a fixed number of ids and keeps the real one. Each shape of
/// hit: ids in the slot, ids in the buffer, the most ids one copy holds and one more. The last
/// word stored is the last record of the buffer, where the copy reads past its ids.
#[test]
fn a_hit_appends_the_ids_it_holds() {
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
        let mut output = vec![Token::from(u32::MAX)];
        hit.append_to(&mut output);
        let expected: Vec<u32> = [u32::MAX].iter().chain(ids).copied().collect();
        assert_eq!(Token::cast_slice(&output), expected, "{word:?}");
    }
}

/// [`WordCache::emit_run`] is a shortcut for [`WordCache::lookup`]. It must write the
/// ids a lookup returns for each word it answers, whether they sit in the slot or in the
/// buffer, and stop at the first word it cannot answer: a word with more ids than
/// [`MAX_RUN_IDS`] ends the first run, a word that was never stored ends the second.
#[test]
fn a_run_writes_what_lookup_returns() {
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
        let mut output = Vec::new();
        let (answered, _) = cache.emit_run(&signatures[next..], |i| words[next + i], &mut output);
        let mut ids = Token::cast_slice(&output).iter();
        for word in &words[next..next + answered] {
            let expected = cache
                .lookup(word)
                .hit()
                .expect("the run answered this word");
            let got: Vec<u32> = ids.by_ref().take(expected.len()).copied().collect();
            assert_eq!(got, expected, "{word:?}");
        }
        assert_eq!(ids.next(), None, "the run wrote ids of no word");
        answers.push(answered);
        next += answered + 1;
    }
    assert_eq!(answers, [5, 1]);
}

/// A word whose home slot is taken is stored in the next slot, and most such words stay there.
/// The run answers it from that slot too: ids in the slot, ids in the buffer, and a word longer
/// than its signature. A one-home table sends every word to slot 0, so the second one stored
/// sits in slot 1.
#[test]
fn a_run_answers_a_word_one_slot_past_its_home() {
    let cases: [(&[u8], &[u32]); 3] = [
        (b"inline", &[7]),
        (b"spilled", &[1, 2, 3, 4]),
        (b"a-word-past-sixteen-bytes", &[5]),
    ];
    for (word, ids) in cases {
        let mut cache = WordCache::new(1);
        store(&mut cache, b"at-home", &[9]);
        store(&mut cache, word, ids);
        let signatures = [Signature::new(word), Signature::new(b"at-home")];
        let words: [&[u8]; 2] = [word, b"at-home"];
        let mut output = Vec::new();
        let (answered, _) = cache.emit_run(&signatures, |i| words[i], &mut output);
        assert_eq!(answered, 2, "{word:?}");
        let expected: Vec<u32> = ids.iter().chain(&[9]).copied().collect();
        assert_eq!(Token::cast_slice(&output), expected, "{word:?}");
    }
}

/// A run that stops on a vacant home slot says so, and the caller then skips the lookup. It
/// must not say so for a taken one: the third word of a one-home table sits two slots on,
/// where the run does not look and a lookup does.
#[test]
fn a_run_reports_a_vacant_home_slot_and_no_other() {
    let run = |cache: &WordCache, words: &[&[u8]]| {
        let signatures: Vec<Signature> = words.iter().map(|word| Signature::new(word)).collect();
        cache.emit_run(&signatures, |i| words[i], &mut Vec::new())
    };
    let mut cache = WordCache::new(1);
    assert_eq!(run(&cache, &[b"third"]), (0, HomeSlot::Vacant));
    for (i, word) in [&b"first"[..], b"second", b"third"].into_iter().enumerate() {
        store(&mut cache, word, &[i as u32]);
    }
    assert_eq!(run(&cache, &[b"third"]), (0, HomeSlot::Unknown));

    // The verdict is about the word the run stopped on, not about one it answered.
    let mut cache = WordCache::new(1 << 8);
    store(&mut cache, b"stored", &[1]);
    assert_eq!(
        run(&cache, &[b"stored", b"never-stored"]),
        (1, HomeSlot::Vacant)
    );
}

/// A word stored through [`WordCache::vacant_home`] is found again, and so is the next word
/// with the same home slot: the first one's slot has to read as taken to the window scan.
#[test]
fn a_word_stored_at_its_vacant_home_leaves_its_window_whole() {
    let mut cache = WordCache::new(1);
    let at = cache.vacant_home(Signature::new(b"first"));
    cache.insert(at, b"first", [1].into_iter());
    store(&mut cache, b"second", &[2]);
    assert_eq!(cache.lookup(b"first").hit(), Some(&[1][..]));
    assert_eq!(cache.lookup(b"second").hit(), Some(&[2][..]));
}

/// A one-home table sends every word to slot 0. The run must not answer a word that has only
/// part of the stored word's signature: ten `a`s against nine (another length), against a
/// last byte changed (another tail), and against a first byte changed (another head). With the
/// ids in the slot and in the buffer, and with the word in its home slot and one slot past it.
#[test]
fn a_run_does_not_answer_a_word_with_part_of_the_signature() {
    for ids in [&[1][..], &[1, 2, 3, 4]] {
        for past_home in [false, true] {
            let mut cache = WordCache::new(1);
            if past_home {
                store(&mut cache, b"at-home", &[9]);
            }
            store(&mut cache, b"aaaaaaaaaa", ids);
            assert_eq!(answered(&cache, &[b"aaaaaaaaaa"]), 1);
            for near in [&b"aaaaaaaaa"[..], b"aaaaaaaaab", b"baaaaaaaaa"] {
                assert_eq!(answered(&cache, &[near]), 0, "{near:?} with {ids:?}");
            }
        }
    }
}

/// Words past [`Signature::BYTES`] with one signature have one home slot. The run answers the
/// one stored there, and the one stored in the slot after it, each with its own ids. It must
/// not answer a third, which neither slot holds, with the ids of one of them.
#[test]
fn a_run_does_not_answer_a_long_word_with_another_words_ids() {
    let mut cache = WordCache::new(1 << 8);
    let [first, second, third]: [&[u8]; 3] = [
        b"01234567-first--89abcdef",
        b"01234567-second-89abcdef",
        b"01234567-third--89abcdef",
    ];
    assert_eq!(Signature::new(first), Signature::new(second));
    assert_eq!(Signature::new(first), Signature::new(third));
    store(&mut cache, first, &[1]);
    assert_eq!(answered(&cache, &[first]), 1);
    assert_eq!(answered(&cache, &[second]), 0);

    store(&mut cache, second, &[2]);
    let signatures = [Signature::new(second), Signature::new(first)];
    let words = [second, first];
    let mut output = Vec::new();
    assert_eq!(cache.emit_run(&signatures, |i| words[i], &mut output).0, 2);
    assert_eq!(Token::cast_slice(&output), [2, 1]);
    assert_eq!(answered(&cache, &[third]), 0);
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

/// A tag is one byte of hash, so about one occupied slot in 255 carries the tag of a word it
/// does not hold. In a one-home table every word starts at slot 0: the first two words found
/// with one tag put that collision on the home slot. The lookup must confirm the signature and
/// keep walking, not trust the tag.
#[test]
fn a_tag_collision_is_confirmed_against_the_signature() {
    let mut cache = WordCache::new(1);
    let mut seen = std::collections::HashMap::new();
    let (first, second) = (0u32..)
        .map(|i| format!("w{i}").into_bytes())
        .find_map(|word| {
            let earlier = seen.insert(tag_of(Signature::new(&word)), word.clone())?;
            Some((earlier, word))
        })
        .expect("255 tags, more words");

    store(&mut cache, &first, &[1]);
    assert_eq!(cache.lookup(&second).hit(), None);
    store(&mut cache, &second, &[2]);
    assert_eq!(cache.lookup(&first).hit(), Some(&[1][..]));
    assert_eq!(cache.lookup(&second).hit(), Some(&[2][..]));
}

/// A word whose whole window is taken is still cached: it evicts its home slot.
/// The other fifteen words in the window keep their ids.
#[test]
fn a_full_window_evicts_only_the_home_slot() {
    // A one-home table: placement_mask is 0, so every word homes at slot 0 and
    // the sixteen slots from there are one shared window.
    let mut cache = WordCache::new(1);
    let words = words_homed_in(&cache, 0..1, WINDOW_SIZE);
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
    cache.spill.set_generation(u32::MAX);
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

    let (mut live, mut in_a_run) = (0, 0);
    for (word, ids) in &expected {
        if let Some(hit) = cache.lookup(word).hit() {
            assert_eq!(hit, &ids[..], "{word:?}");
            live += 1;
        }
        // The run is a shortcut for the lookup: what it answers, it answers the same.
        let mut output = Vec::new();
        if cache
            .emit_run(&[Signature::new(word)], |_| word, &mut output)
            .0
            == 1
        {
            assert_eq!(Token::cast_slice(&output), &ids[..], "{word:?}");
            in_a_run += 1;
        }
    }
    assert!(live > 0, "everything was evicted, the test proves nothing");
    assert!(
        in_a_run > 0,
        "the run answered no word, the test proves nothing"
    );
}
