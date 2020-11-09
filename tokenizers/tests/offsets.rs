mod common;

use common::*;
use tokenizers::tokenizer::AddedToken;

macro_rules! check_offsets {
    ($input: expr, $output:expr, $offset:expr, $result:expr) => {
        let offsets = $output.get_offsets()[$offset];
        assert_eq!(&$input[offsets.0..offsets.1], $result);
    };
}

#[test]
fn byte_level_basic() {
    // Without trimming offsets
    let tokenizer = get_byte_level(true, false);

    let input = "Hello there, how are you?";
    let output = tokenizer.encode(input, false).unwrap();

    check_offsets!(input, output, 0, "Hello");
    check_offsets!(input, output, 1, " there");
    check_offsets!(input, output, 2, ",");
    check_offsets!(input, output, 3, " how");
    check_offsets!(input, output, 4, " are");
    check_offsets!(input, output, 5, " you");
    check_offsets!(input, output, 6, "?");

    // And when trimming offsets:
    let tokenizer = get_byte_level(true, true);

    let input = "Hello there, how are you?";
    let output = tokenizer.encode(input, false).unwrap();

    check_offsets!(input, output, 0, "Hello");
    check_offsets!(input, output, 1, "there");
    check_offsets!(input, output, 2, ",");
    check_offsets!(input, output, 3, "how");
    check_offsets!(input, output, 4, "are");
    check_offsets!(input, output, 5, "you");
    check_offsets!(input, output, 6, "?");
}

#[test]
fn byte_level_unicode() {
    let tokenizer = get_byte_level(true, false);

    let input = "i⭢j";
    let output = tokenizer.encode(input, false).unwrap();

    check_offsets!(input, output, 1, "⭢");
    check_offsets!(input, output, 2, "⭢");
    check_offsets!(input, output, 3, "⭢");
}

#[test]
fn byte_level_double_sequence() {
    let input_a = "My name is Anthony";
    let input_b = "What is my name?";

    // Without trimming offsets
    let tokenizer = get_byte_level(true, false);
    let output = tokenizer.encode((input_a, input_b), false).unwrap();

    let offsets = output.get_offsets();
    assert_eq!(
        offsets,
        &[
            (0, 2),
            (2, 7),
            (7, 10),
            (10, 18),
            (0, 4),
            (4, 7),
            (7, 10),
            (10, 15),
            (15, 16)
        ]
    );
    assert_eq!(
        output.get_word_ids(),
        &[
            Some(0),
            Some(1),
            Some(2),
            Some(3),
            Some(0),
            Some(1),
            Some(2),
            Some(3),
            Some(4)
        ]
    );
    assert_eq!(output.get_type_ids(), &[0, 0, 0, 0, 1, 1, 1, 1, 1]);

    // When trimming offsets
    let tokenizer = get_byte_level(true, true);
    let output = tokenizer.encode((input_a, input_b), false).unwrap();
    let offsets = output.get_offsets();
    assert_eq!(
        offsets,
        &[
            (0, 2),
            (3, 7),
            (8, 10),
            (11, 18),
            (0, 4),
            (5, 7),
            (8, 10),
            (11, 15),
            (15, 16)
        ]
    );
}

#[test]
fn byte_level_pre_tokenized_sequence() {
    let input = ["My", "name", "is", "Anthonino"];

    // Without trimming offsets
    let tokenizer = get_byte_level(true, false);
    let output = tokenizer.encode(&input[..], false).unwrap();

    assert_eq!(
        output.get_tokens(),
        &["ĠMy", "Ġname", "Ġis", "ĠAnth", "on", "ino"]
    );
    assert_eq!(
        output.get_word_ids(),
        &[Some(0), Some(1), Some(2), Some(3), Some(3), Some(3)]
    );
    assert_eq!(
        output.get_offsets(),
        &[(0, 2), (0, 4), (0, 2), (0, 4), (4, 6), (6, 9)]
    );
}

#[test]
#[ignore]
fn byte_level_pre_tokenized_sequence_with_trimming() {
    let input = ["My", "name", "is", "Anthonino"];

    // When trimming offsets (expect same result)
    let tokenizer = get_byte_level(true, true);
    let output = tokenizer.encode(&input[..], false).unwrap();

    assert_eq!(
        output.get_word_ids(),
        &[Some(0), Some(1), Some(2), Some(3), Some(3), Some(3)]
    );
    assert_eq!(
        output.get_offsets(),
        &[(0, 2), (0, 4), (0, 2), (0, 4), (4, 6), (6, 9)]
    );
}

#[test]
fn split_on_added_tokens_bert() {
    let input = "Yesterday I saw a [MASK] far away";

    let mut tokenizer = get_bert();
    tokenizer.add_special_tokens(&[AddedToken::from("[MASK]", true)]);
    let output = tokenizer.encode(input, false).unwrap();

    assert_eq!(
        output.get_offsets(),
        &[
            (0, 9),
            (10, 11),
            (12, 15),
            (16, 17),
            (18, 24),
            (25, 28),
            (29, 33)
        ]
    );
    assert_eq!(
        output.get_tokens(),
        &["yesterday", "i", "saw", "a", "[MASK]", "far", "away"]
    );
    assert_eq!(
        output.get_word_ids(),
        &[
            Some(0),
            Some(1),
            Some(2),
            Some(3),
            Some(4),
            Some(5),
            Some(6)
        ]
    );
}
