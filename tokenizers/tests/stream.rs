use tokenizers::{
    normalizers,
    parallelism::{get_parallelism, set_parallelism},
    pre_tokenizers::split::{Split, SplitPattern},
    AddedToken, NormalizerWrapper, PreTokenizerWrapper, SplitDelimiterBehavior, Tokenizer,
};

#[cfg(feature = "pcre2")]
use std::sync::{LazyLock, Mutex};

#[cfg(feature = "pcre2")]
static PARALLELISM_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

#[cfg(feature = "pcre2")]
fn with_parallelism<T>(parallel: bool, f: impl FnOnce() -> T) -> T {
    let previous = get_parallelism();
    set_parallelism(parallel);
    let result = f();
    set_parallelism(previous);
    result
}

#[test]
fn test_decoding_with_added_bpe() {
    let mut tokenizer = Tokenizer::from_file("data/llama-3-tokenizer.json").unwrap();
    tokenizer.with_normalizer(Some(NormalizerWrapper::from(normalizers::ByteLevel::new())));
    tokenizer.with_pre_tokenizer(Some(PreTokenizerWrapper::Split(
        Split::new(
            SplitPattern::Regex(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+".into()),
            SplitDelimiterBehavior::Isolated,
            false,
        )
        .unwrap(),
    )));
    tokenizer.add_tokens(&[AddedToken::from("嗎", false).normalized(false)]);
    let encoded = tokenizer
        .encode("Hey! how is this token: 嗎", false)
        .unwrap();
    assert_eq!(
        encoded.get_ids(),
        [19182, 0, 1268, 602, 82, 62428, 82, 4037, 25, 220, 128256]
    );
    assert_eq!(
        encoded.get_tokens(),
        ["Hey", "!", "Ġhow", "Ġi", "s", "Ġthi", "s", "Ġtoken", ":", "Ġ", "嗎"]
    );

    let decoded = tokenizer.decode(encoded.get_ids(), false);
    assert_eq!(decoded.unwrap(), "Hey! how is this token: 嗎");

    tokenizer.add_tokens(&[AddedToken::from("д", false).normalized(true)]);
    let encoded = tokenizer
        .encode("Hey! how is this token: д", false)
        .unwrap();
    assert_eq!(
        encoded.get_ids(),
        [19182, 0, 1268, 602, 82, 62428, 82, 4037, 25, 220, 128257]
    );
    assert_eq!(
        encoded.get_tokens(),
        ["Hey", "!", "Ġhow", "Ġi", "s", "Ġthi", "s", "Ġtoken", ":", "Ġ", "Ð´"]
    );
    let decoded = tokenizer.decode(encoded.get_ids(), false);
    assert_eq!(decoded.unwrap(), "Hey! how is this token: д")
}

#[test]
fn test_decode_stream_step_no_panic() {
    let tokenizer = Tokenizer::from_file("data/llama-3-tokenizer.json").unwrap();

    // "A B C D E F G H I J"
    let mut decode_stream = tokenizer.decode_stream(false);
    assert_eq!(decode_stream.step(32).unwrap(), Some("A".to_string()));
    assert_eq!(decode_stream.step(426).unwrap(), Some(" B".to_string()));
    assert_eq!(decode_stream.step(356).unwrap(), Some(" C".to_string()));
    assert_eq!(decode_stream.step(423).unwrap(), Some(" D".to_string()));
    assert_eq!(decode_stream.step(469).unwrap(), Some(" E".to_string()));
    assert_eq!(decode_stream.step(435).unwrap(), Some(" F".to_string()));
    assert_eq!(decode_stream.step(480).unwrap(), Some(" G".to_string()));
    assert_eq!(decode_stream.step(473).unwrap(), Some(" H".to_string()));
    assert_eq!(decode_stream.step(358).unwrap(), Some(" I".to_string()));
    assert_eq!(decode_stream.step(622).unwrap(), Some(" J".to_string()));
    // for (i, &token) in output_tokens.iter().enumerate() {}

    // "삥뽕빵" (Korean words composed of 2-3 tokens: [80690, 98], [167, 121, 243], and [102457, 113])
    let mut decode_stream = tokenizer.decode_stream(false);
    assert_eq!(decode_stream.step(80690).unwrap(), None);
    assert_eq!(decode_stream.step(98).unwrap(), Some("삥".to_string()));
    assert_eq!(decode_stream.step(167).unwrap(), None);
    assert_eq!(decode_stream.step(121).unwrap(), None);
    assert_eq!(decode_stream.step(243).unwrap(), Some("뽕".to_string()));
    assert_eq!(decode_stream.step(102457).unwrap(), None);
    assert_eq!(decode_stream.step(113).unwrap(), Some("빵".to_string()));
}

#[cfg(feature = "pcre2")]
#[test]
fn test_long_context_encode_matches_sequential() {
    let _guard = PARALLELISM_LOCK.lock().unwrap();
    let tokenizer = Tokenizer::from_file("data/llama-3-tokenizer.json").unwrap();
    let chunk = "Hello, y'all! How are you 😁 ? 12345 -- Καλημέρα.\n";
    let input = chunk.repeat(600);
    assert!(input.len() > 16_384);

    let sequential = with_parallelism(false, || tokenizer.encode(input.as_str(), false).unwrap());
    let parallel = with_parallelism(true, || tokenizer.encode(input.as_str(), false).unwrap());

    assert_eq!(sequential.get_ids(), parallel.get_ids());
    assert_eq!(sequential.get_tokens(), parallel.get_tokens());
}

#[cfg(feature = "pcre2")]
#[test]
fn test_long_context_char_offsets_match_sequential() {
    let _guard = PARALLELISM_LOCK.lock().unwrap();
    let tokenizer = Tokenizer::from_file("data/llama-3-tokenizer.json").unwrap();
    let chunk = "Hello, y'all! How are you 😁 ? 12345 -- Καλημέρα.\n";
    let input = chunk.repeat(600);
    assert!(input.len() > 16_384);

    let sequential = with_parallelism(false, || {
        tokenizer
            .encode_char_offsets(input.as_str(), false)
            .unwrap()
    });
    let parallel = with_parallelism(true, || {
        tokenizer
            .encode_char_offsets(input.as_str(), false)
            .unwrap()
    });

    assert_eq!(sequential.get_ids(), parallel.get_ids());
    assert_eq!(sequential.get_tokens(), parallel.get_tokens());
    assert_eq!(sequential.get_offsets(), parallel.get_offsets());
}
