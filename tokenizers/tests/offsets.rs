use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::tokenizer::{get_range_of, EncodeInput, Tokenizer};

fn get_byte_level(add_prefix_space: bool, trim_offsets: bool) -> Tokenizer {
    let mut tokenizer = Tokenizer::new(Box::new(
        BPE::from_files("data/gpt2-vocab.json", "data/gpt2-merges.txt")
            .build()
            .expect("Files not found, run `make test` to download these files"),
    ));
    tokenizer.with_pre_tokenizer(Box::new(
        ByteLevel::default().add_prefix_space(add_prefix_space),
    ));
    tokenizer.with_decoder(Box::new(ByteLevel::default()));
    tokenizer.with_post_processor(Box::new(ByteLevel::default().trim_offsets(trim_offsets)));

    tokenizer
}

#[inline]
fn offset_as_range(offset: (usize, usize)) -> std::ops::Range<usize> {
    offset.0..offset.1
}

#[test]
fn byte_level_basic() {
    // Without trimming offsets
    let tokenizer = get_byte_level(true, false);

    let input = String::from("Hello there, how are you?");
    let output = tokenizer
        .encode(EncodeInput::Single(input.clone()), false)
        .unwrap();

    let offsets = output.get_offsets();
    assert_eq!(
        get_range_of(&input, offset_as_range(offsets[0])),
        Some("Hello")
    );
    assert_eq!(
        get_range_of(&input, offset_as_range(offsets[1])),
        Some(" there")
    );
    assert_eq!(get_range_of(&input, offset_as_range(offsets[2])), Some(","));
    assert_eq!(
        get_range_of(&input, offset_as_range(offsets[3])),
        Some(" how")
    );
    assert_eq!(
        get_range_of(&input, offset_as_range(offsets[4])),
        Some(" are")
    );
    assert_eq!(
        get_range_of(&input, offset_as_range(offsets[5])),
        Some(" you")
    );
    assert_eq!(get_range_of(&input, offset_as_range(offsets[6])), Some("?"));

    // And when trimming offsets:
    let tokenizer = get_byte_level(true, true);

    let input = String::from("Hello there, how are you?");
    let output = tokenizer
        .encode(EncodeInput::Single(input.clone()), false)
        .unwrap();

    let offsets = output.get_offsets();
    assert_eq!(
        get_range_of(&input, offset_as_range(offsets[0])),
        Some("Hello")
    );
    assert_eq!(
        get_range_of(&input, offset_as_range(offsets[1])),
        Some("there")
    );
    assert_eq!(get_range_of(&input, offset_as_range(offsets[2])), Some(","));
    assert_eq!(
        get_range_of(&input, offset_as_range(offsets[3])),
        Some("how")
    );
    assert_eq!(
        get_range_of(&input, offset_as_range(offsets[4])),
        Some("are")
    );
    assert_eq!(
        get_range_of(&input, offset_as_range(offsets[5])),
        Some("you")
    );
    assert_eq!(get_range_of(&input, offset_as_range(offsets[6])), Some("?"));
}

#[test]
fn byte_level_unicode() {
    let tokenizer = get_byte_level(true, false);

    let input = String::from("i⭢j");
    let output = tokenizer
        .encode(EncodeInput::Single(input.clone()), false)
        .unwrap();

    let offsets = output.get_offsets();
    assert_eq!(get_range_of(&input, offset_as_range(offsets[1])), Some("⭢"));
    assert_eq!(get_range_of(&input, offset_as_range(offsets[2])), Some("⭢"));
    assert_eq!(get_range_of(&input, offset_as_range(offsets[3])), Some("⭢"));
}

#[test]
fn byte_level_double_sequence() {
    let input_a = String::from("My name is Anthony");
    let input_b = String::from("What is my name?");

    // Without trimming offsets
    let tokenizer = get_byte_level(true, false);
    let output = tokenizer
        .encode(EncodeInput::Dual(input_a.clone(), input_b.clone()), false)
        .unwrap();

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
    let output = tokenizer
        .encode(EncodeInput::Dual(input_a, input_b), false)
        .unwrap();
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
