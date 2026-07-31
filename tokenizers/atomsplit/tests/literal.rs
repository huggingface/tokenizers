//! Tests for the literal search. Kept out of `src/` so the core stays production-only.
use atomsplit::literal::{EmptyPattern, Literal};

#[test]
fn finds_every_match_and_nothing_else() {
    let literal = Literal::new(b"-").unwrap();
    assert_eq!(literal.matches(b"a-b--c").collect::<Vec<_>>(), [1, 3, 4]);
    assert_eq!(literal.matches(b"none here").count(), 0);
    assert_eq!(literal.matches(b"").count(), 0);
    assert_eq!(literal.pattern(), b"-");
}

#[test]
fn a_multi_byte_pattern_only_matches_whole() {
    // `▁` is U+2581 = E2 96 81. The other characters here share its first byte and nothing else, so a
    // search for that byte alone would report all of them.
    let literal = Literal::new("▁".as_bytes()).unwrap();
    let text = "a—b“c…d▁e";
    assert_eq!(
        literal.matches(text.as_bytes()).collect::<Vec<_>>(),
        [text.find('▁').unwrap()]
    );
}

#[test]
fn matches_do_not_overlap() {
    let literal = Literal::new(b"aa").unwrap();
    assert_eq!(literal.matches(b"aaa").collect::<Vec<_>>(), [0]);
    assert_eq!(literal.matches(b"aaaa").collect::<Vec<_>>(), [0, 2]);
}

/// A `Literal` is stored inline in the normalizer and decoder enums, where every other variant is a
/// handful of bytes. `memmem::Finder` itself is a few hundred, so it has to stay behind a pointer.
#[test]
fn a_literal_is_pointer_sized() {
    assert_eq!(
        size_of::<Literal>(),
        size_of::<*const u8>(),
        "a Literal must not carry its finder inline"
    );
}

#[test]
fn an_empty_pattern_is_rejected() {
    assert_eq!(Literal::new(b"").unwrap_err(), EmptyPattern);
    assert_eq!(
        EmptyPattern.to_string(),
        "an empty pattern matches everywhere"
    );
}

// ---- the batch scan, `matches_into` ----

/// Run `matches_into` with a buffer of exactly the documented size and return what it wrote.
fn batch(literal: &Literal, text: &[u8]) -> Vec<u32> {
    let mut out = vec![0u32; text.len() / literal.pattern().len() + 4];
    let count = literal.matches_into(text, &mut out);
    out.truncate(count);
    out
}

fn iterated(literal: &Literal, text: &[u8]) -> Vec<u32> {
    literal.matches(text).map(|p| p as u32).collect()
}

#[test]
fn matches_into_reports_the_iterator_positions() {
    let dash = Literal::new(b"-").unwrap();
    assert_eq!(batch(&dash, b"a-b--c"), [1, 3, 4]);
    assert_eq!(batch(&dash, b"-starts and ends-"), [0, 16]);
    assert_eq!(batch(&dash, b"none here"), []);
    assert_eq!(batch(&dash, b""), []);

    let space = Literal::new(b" ").unwrap();
    let text = b"the quick brown fox jumps over the lazy dog".repeat(40);
    assert_eq!(batch(&space, &text), iterated(&space, &text));

    let metaspace = Literal::new("\u{2581}".as_bytes()).unwrap();
    let text = "\u{2581}the\u{2581}quick\u{2581}brown\u{2581}fox".repeat(40);
    assert_eq!(
        batch(&metaspace, text.as_bytes()),
        iterated(&metaspace, text.as_bytes())
    );
}

/// The scan works in blocks; a match starting on either side of an internal block edge, or on
/// the last position where the pattern still fits, must not be missed or doubled.
#[test]
fn matches_into_is_seamless_across_block_boundaries() {
    for pattern in [&b"-"[..], "\u{2581}".as_bytes()] {
        let literal = Literal::new(pattern).unwrap();
        for start in [0, 13, 14, 15, 16, 17, 61, 62, 63, 64, 65, 127, 128] {
            for len in [start + pattern.len(), 90, 129, 200] {
                if start + pattern.len() > len {
                    continue;
                }
                let mut text = vec![b'x'; len];
                text[start..start + pattern.len()].copy_from_slice(pattern);
                assert_eq!(
                    batch(&literal, &text),
                    [start as u32],
                    "pattern {pattern:?} at {start} in {len} bytes"
                );
            }
        }
    }
}

/// Every position matches: the largest count a text can produce, written into a buffer of
/// exactly the documented size.
#[test]
fn matches_into_handles_the_densest_text() {
    let a = Literal::new(b"a").unwrap();
    let text = vec![b'a'; 300];
    assert_eq!(batch(&a, &text), (0..300).collect::<Vec<u32>>());

    let metaspace = Literal::new("\u{2581}".as_bytes()).unwrap();
    let text = "\u{2581}".repeat(100);
    assert_eq!(
        batch(&metaspace, text.as_bytes()),
        (0..100).map(|i| i * 3).collect::<Vec<u32>>()
    );
}

#[test]
fn a_self_overlapping_pattern_keeps_non_overlapping_matches() {
    let aa = Literal::new(b"aa").unwrap();
    assert_eq!(batch(&aa, b"aaa"), [0]);
    let run = [b'a'; 100];
    assert_eq!(batch(&aa, &run), iterated(&aa, &run));

    let aba = Literal::new(b"aba").unwrap();
    assert_eq!(batch(&aba, b"ababa"), [0]);
}

#[test]
fn a_pattern_longer_than_three_bytes_still_matches() {
    let literal = Literal::new(b"abcd").unwrap();
    let text = b"abcd xabcdx abcabcd".repeat(20);
    assert_eq!(batch(&literal, &text), iterated(&literal, &text));
}

#[test]
#[should_panic(expected = "matches_into")]
fn an_undersized_buffer_is_rejected() {
    let a = Literal::new(b"a").unwrap();
    let mut out = vec![0u32; 7]; // "aaaa" needs 4 / 1 + 4 = 8
    a.matches_into(b"aaaa", &mut out);
}

// ---- the streaming pair, `count_matches` + `for_each_match` ----

fn streamed(literal: &Literal, text: &[u8]) -> Vec<u32> {
    let mut out = Vec::new();
    literal.for_each_match(text, |start| out.push(start as u32));
    out
}

/// The streaming scan works through a fixed window; a text much longer than any plausible
/// window must report the same matches as the iterator, including around every window edge.
#[test]
fn for_each_match_streams_the_iterator_offsets_across_windows() {
    for pattern in [&b" "[..], "\u{2581}".as_bytes(), b"ab"] {
        let literal = Literal::new(pattern).unwrap();
        let mut text = b"word ".repeat(4_000); // ~20KB, matches every 5 bytes
        text.extend_from_slice("\u{2581}ab".repeat(2_000).as_bytes());
        assert_eq!(streamed(&literal, &text), iterated(&literal, &text));
        assert_eq!(
            literal.count_matches(&text),
            literal.matches(&text).count(),
            "count for {pattern:?}"
        );
    }
}

/// The densest possible long text: every byte starts a match, in every window.
#[test]
fn for_each_match_handles_dense_windows() {
    let a = Literal::new(b"a").unwrap();
    let text = vec![b'a'; 10_000];
    assert_eq!(streamed(&a, &text), (0..10_000).collect::<Vec<u32>>());
    assert_eq!(a.count_matches(&text), 10_000);
}

#[test]
fn for_each_match_takes_the_iterator_route_for_uncovered_patterns() {
    // Self-overlapping and longer-than-three patterns are exactly the ones the batch scan
    // refuses; the streaming pair must still report the iterator's non-overlapping matches.
    for pattern in [&b"aa"[..], b"aba", b"abcd"] {
        let literal = Literal::new(pattern).unwrap();
        let text = b"aabaabcdaaaba".repeat(2_000);
        assert_eq!(streamed(&literal, &text), iterated(&literal, &text));
        assert_eq!(literal.count_matches(&text), literal.matches(&text).count());
    }
}

#[test]
fn streaming_agrees_with_the_iterator_on_random_lengths() {
    let mut state = 0x1234_5678_9ABC_DEF0u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let patterns: &[&[u8]] = &[b"a", b"ab", b"aa", b"aba", b"abc", b"abca"];
    for _ in 0..500 {
        // lengths spread around plausible window sizes, so edges get hit both exactly and off by one
        let len = (next() % 8_000) as usize;
        let text: Vec<u8> = (0..len).map(|_| b'a' + (next() % 3) as u8).collect();
        let literal = Literal::new(patterns[(next() % patterns.len() as u64) as usize]).unwrap();
        assert_eq!(
            streamed(&literal, &text),
            iterated(&literal, &text),
            "pattern {:?} len {len}",
            literal.pattern()
        );
        assert_eq!(literal.count_matches(&text), literal.matches(&text).count());
    }
}

#[test]
fn matches_into_agrees_with_the_iterator_on_random_inputs() {
    // xorshift64: deterministic, no dev-dependency needed.
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let patterns: &[&[u8]] = &[
        b"a", b"b", b"ab", b"aa", b"ba", b"aab", b"aba", b"abc", b"abca",
    ];
    for _ in 0..20_000 {
        let len = (next() % 300) as usize;
        let alphabet = 2 + (next() % 2) as u8;
        let text: Vec<u8> = (0..len)
            .map(|_| b'a' + (next() % alphabet as u64) as u8)
            .collect();
        let literal = Literal::new(patterns[(next() % patterns.len() as u64) as usize]).unwrap();
        assert_eq!(
            batch(&literal, &text),
            iterated(&literal, &text),
            "pattern {:?} text {:?}",
            literal.pattern(),
            text
        );
    }
}
