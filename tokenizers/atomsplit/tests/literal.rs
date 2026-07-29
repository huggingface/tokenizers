//! Tests for the literal search. Kept out of `src/` so the core stays production-only.
use atomsplit::literal::Literal;

#[test]
fn finds_every_match_and_nothing_else() {
    let literal = Literal::new(b"-");
    assert_eq!(literal.matches(b"a-b--c").collect::<Vec<_>>(), [1, 3, 4]);
    assert_eq!(literal.matches(b"none here").count(), 0);
    assert_eq!(literal.matches(b"").count(), 0);
    assert_eq!(literal.pattern(), b"-");
}

#[test]
fn a_multi_byte_pattern_only_matches_whole() {
    // `▁` is U+2581 = E2 96 81. The other characters here share its first byte and nothing else, so a
    // search for that byte alone would report all of them.
    let literal = Literal::new("▁".as_bytes());
    let text = "a—b“c…d▁e";
    assert_eq!(
        literal.matches(text.as_bytes()).collect::<Vec<_>>(),
        [text.find('▁').unwrap()]
    );
}

#[test]
fn matches_do_not_overlap() {
    assert_eq!(Literal::new(b"aa").matches(b"aaa").collect::<Vec<_>>(), [0]);
    assert_eq!(
        Literal::new(b"aa").matches(b"aaaa").collect::<Vec<_>>(),
        [0, 2]
    );
}

#[test]
#[should_panic(expected = "empty pattern")]
fn an_empty_pattern_is_rejected() {
    let _ = Literal::new(b"");
}
