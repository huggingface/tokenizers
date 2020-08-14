mod common;

use common::*;
use tokenizers::tokenizer::AddedToken;

#[test]
fn add_tokens() {
    let mut tokenizer = get_empty();

    assert_eq!(
        tokenizer.add_special_tokens(&[
            AddedToken::from("<cls>", true),
            AddedToken::from("<sep>", true)
        ]),
        2
    );
    assert_eq!(tokenizer.token_to_id("<cls>"), Some(0));
    assert_eq!(tokenizer.token_to_id("<sep>"), Some(1));

    assert_eq!(
        tokenizer.add_tokens(&[
            AddedToken::from("hello", false),
            AddedToken::from("world", false)
        ]),
        2
    );
    assert_eq!(tokenizer.token_to_id("hello"), Some(2));
    assert_eq!(tokenizer.token_to_id("world"), Some(3));
}

#[test]
fn lstrip_tokens() {
    let mut tokenizer = get_byte_level(true, false);
    tokenizer.add_special_tokens(&[AddedToken::from("<mask>", true).lstrip(true)]);

    let input = "I saw a <mask> 😺";
    let output = tokenizer.encode(input, false).unwrap();

    assert_eq!(
        output.get_tokens(),
        &["ĠI", "Ġsaw", "Ġa", " <mask>", "ĠðŁĺ", "º"]
    );
    assert_eq!(
        output.get_offsets(),
        &[(0, 1), (1, 5), (5, 7), (7, 14), (14, 19), (15, 19)]
    );
}

#[test]
fn rstrip_tokens() {
    let mut tokenizer = get_byte_level(false, false);
    tokenizer.add_special_tokens(&[AddedToken::from("<mask>", true).rstrip(true)]);

    let input = "I saw a <mask> 😺";
    let output = tokenizer.encode(input, false).unwrap();

    assert_eq!(
        output.get_tokens(),
        &["I", "Ġsaw", "Ġa", "Ġ", "<mask> ", "ðŁĺ", "º"]
    );

    // When `add_prefix_space = true` rstrip cannot work as a prefix space is added
    // to the next token
    let mut tokenizer = get_byte_level(true, false);
    tokenizer.add_special_tokens(&[AddedToken::from("<mask>", true).rstrip(true)]);

    let input = "I saw a <mask> 😺";
    let output = tokenizer.encode(input, false).unwrap();

    assert_eq!(
        output.get_tokens(),
        &["ĠI", "Ġsaw", "Ġa", "Ġ", "<mask> ", "ĠðŁĺ", "º"]
    );
}

#[test]
fn single_word_tokens() {
    // If `single_word = true` it shouldn't split `dancing`
    let mut tokenizer = get_byte_level(false, false);
    tokenizer.add_special_tokens(&[AddedToken::from("ing", true).single_word(true)]);

    let input = "I like dancing";
    let output = tokenizer.encode(input, false).unwrap();

    assert_eq!(output.get_tokens(), &["I", "Ġlike", "Ġdancing"]);

    // If `single_word = false` it should split `dancing`
    let mut tokenizer = get_byte_level(false, false);
    tokenizer.add_special_tokens(&[AddedToken::from("ing", true).single_word(false)]);

    let input = "I like dancing";
    let output = tokenizer.encode(input, false).unwrap();

    assert_eq!(output.get_tokens(), &["I", "Ġlike", "Ġd", "anc", "ing"]);
}

#[test]
fn overlapping_tokens() {
    let mut tokenizer = get_byte_level(false, false);

    tokenizer.add_special_tokens(&[AddedToken::from("danc", true)]);
    tokenizer.add_special_tokens(&[AddedToken::from("nci", true)]);
    tokenizer.add_special_tokens(&[AddedToken::from("ing", true)]);

    let input = "I like dancing";
    let output = tokenizer.encode(input, false).unwrap();

    assert_eq!(output.get_tokens(), &["I", "Ġlike", "Ġ", "danc", "ing"]);

    let mut tokenizer = get_byte_level(false, false);

    tokenizer.add_special_tokens(&[AddedToken::from("nci", true)]);
    tokenizer.add_special_tokens(&[AddedToken::from("danc", true)]);
    tokenizer.add_special_tokens(&[AddedToken::from("ing", true)]);
    tokenizer.add_special_tokens(&[AddedToken::from("ike", true)]);

    let output = tokenizer.encode(input, false).unwrap();

    assert_eq!(output.get_tokens(), &["I", "Ġl", "ike", "Ġda", "nci", "ng"]);
}
