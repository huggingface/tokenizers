mod common;

use common::*;
use tokenizers::tokenizer::AddedToken;

#[test]
fn add_tokens() {
    let mut tokenizer = get_empty();

    assert_eq!(
        tokenizer
            .add_special_tokens([
                AddedToken::from("<cls>", true),
                AddedToken::from("<sep>", true)
            ])
            .unwrap(),
        2
    );
    assert_eq!(tokenizer.token_to_id("<cls>"), Some(0));
    assert_eq!(tokenizer.token_to_id("<sep>"), Some(1));

    assert_eq!(
        tokenizer
            .add_tokens([
                AddedToken::from("hello", false),
                AddedToken::from("world", false)
            ])
            .unwrap(),
        2
    );
    assert_eq!(tokenizer.token_to_id("hello"), Some(2));
    assert_eq!(tokenizer.token_to_id("world"), Some(3));
}

#[test]
fn lstrip_tokens() {
    let mut tokenizer = get_byte_level(true, false);
    tokenizer
        .add_special_tokens([AddedToken::from("<mask>", true).lstrip(true)])
        .unwrap();

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
    tokenizer
        .add_special_tokens([AddedToken::from("<mask>", true).rstrip(true)])
        .unwrap();

    let input = "I saw a <mask> 😺";
    let output = tokenizer.encode(input, false).unwrap();

    assert_eq!(
        output.get_tokens(),
        &["I", "Ġsaw", "Ġa", "Ġ", "<mask> ", "ðŁĺ", "º"]
    );

    // When `add_prefix_space = true` rstrip cannot work as a prefix space is added
    // to the next token
    let mut tokenizer = get_byte_level(true, false);
    tokenizer
        .add_special_tokens([AddedToken::from("<mask>", true).rstrip(true)])
        .unwrap();

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
    tokenizer
        .add_special_tokens([AddedToken::from("ing", true).single_word(true)])
        .unwrap();

    let input = "I like dancing";
    let output = tokenizer.encode(input, false).unwrap();

    assert_eq!(output.get_tokens(), &["I", "Ġlike", "Ġdancing"]);

    // If `single_word = false` it should split `dancing`
    let mut tokenizer = get_byte_level(false, false);
    tokenizer
        .add_special_tokens([AddedToken::from("ing", true).single_word(false)])
        .unwrap();

    let input = "I like dancing";
    let output = tokenizer.encode(input, false).unwrap();

    assert_eq!(output.get_tokens(), &["I", "Ġlike", "Ġd", "anc", "ing"]);
}

#[test]
fn overlapping_tokens() {
    let mut tokenizer = get_byte_level(false, false);

    tokenizer
        .add_special_tokens([AddedToken::from("danc", true)])
        .unwrap();
    tokenizer
        .add_special_tokens([AddedToken::from("nci", true)])
        .unwrap();
    tokenizer
        .add_special_tokens([AddedToken::from("ing", true)])
        .unwrap();

    let input = "I like dancing";
    let output = tokenizer.encode(input, false).unwrap();

    assert_eq!(output.get_tokens(), &["I", "Ġlike", "Ġ", "danc", "ing"]);

    let mut tokenizer = get_byte_level(false, false);

    tokenizer
        .add_special_tokens([AddedToken::from("nci", true)])
        .unwrap();
    tokenizer
        .add_special_tokens([AddedToken::from("danc", true)])
        .unwrap();
    tokenizer
        .add_special_tokens([AddedToken::from("ing", true)])
        .unwrap();
    tokenizer
        .add_special_tokens([AddedToken::from("ike", true)])
        .unwrap();

    let output = tokenizer.encode(input, false).unwrap();

    // Breaking change but following `transformers` breaking change.
    // This behavior is deemed not used in practice:
    // https://github.com/huggingface/transformers/pull/13220
    // Order does NOT matter. (We could make it work again but the trie
    // would need to keep insertion order too)
    //
    // assert_eq!(output.get_tokens(), &["I", "Ġlike", "Ġda", "nci", "ng"]);
    assert_eq!(output.get_tokens(), &["I", "Ġl", "ike", "Ġ", "danc", "ing"]);
}
