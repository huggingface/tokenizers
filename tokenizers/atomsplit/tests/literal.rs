//! Tests for the literal search. Kept out of `src/` so the core stays production-only.
use atomsplit::literal::{InvalidPattern, Literal};

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

#[test]
fn finds_matches_beyond_the_first_simd_load() {
    // The matcher reads the text 32 bytes at a time; a match in the second load checks that
    // reported offsets count from the start of the text, not the start of the load.
    let mut text = vec![b'x'; 40];
    text.extend_from_slice(b"-yy");
    let literal = Literal::new(b"-").unwrap();
    assert_eq!(literal.matches(&text).collect::<Vec<_>>(), [40]);
}

#[test]
fn finds_a_match_straddling_two_simd_loads() {
    // "ab" placed so that 'a' is the last byte of the first 32-byte load and 'b' the first byte
    // of the second: the match is invisible to either load alone.
    let mut text = vec![b'x'; 31];
    text.extend_from_slice(b"ab");
    text.extend_from_slice(&[b'x'; 10]);
    let literal = Literal::new(b"ab").unwrap();
    assert_eq!(literal.matches(&text).collect::<Vec<_>>(), [31]);
}

#[test]
fn finds_matches_at_every_load_alignment() {
    // 120 bytes of "xab" put a match at every offset of the form 3k + 1, so some match starts at
    // every position relative to a 32-byte load, including each straddle of a load boundary.
    let text = b"xab".repeat(40);
    let literal = Literal::new(b"ab").unwrap();
    let expected: Vec<usize> = (0..40).map(|k| 3 * k + 1).collect();
    assert_eq!(literal.matches(&text).collect::<Vec<_>>(), expected);
}

#[test]
fn an_empty_pattern_is_rejected() {
    assert_eq!(Literal::new(b"").unwrap_err(), InvalidPattern::Empty);
    assert_eq!(
        InvalidPattern::Empty.to_string(),
        "an empty pattern matches everywhere"
    );
}

#[test]
fn patterns_longer_than_max_len_are_rejected() {
    assert!(Literal::new(&[b'a'; Literal::MAX_PATTERN_LEN]).is_ok());
    assert_eq!(
        Literal::new(&[b'a'; Literal::MAX_PATTERN_LEN + 1]).unwrap_err(),
        InvalidPattern::TooLong
    );
    assert_eq!(
        InvalidPattern::TooLong.to_string(),
        "patterns longer than 32 bytes are not supported"
    );
}
