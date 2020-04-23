mod common;

use common::*;
use tokenizers::tokenizer::{get_range_of, AddedToken};

macro_rules! check_offsets {
    ($input: expr, $output:expr, $offset:expr, $result:expr) => {
        let offsets = $output.get_offsets()[$offset];
        assert_eq!(get_range_of(&$input, offsets.0..offsets.1), Some($result));
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
fn split_on_added_tokens_bert() {
    let input = "Yesterday I saw a [MASK] far away";

    let mut tokenizer = get_bert();
    tokenizer.add_special_tokens(&[AddedToken::from("[MASK]".into())]);
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
}
