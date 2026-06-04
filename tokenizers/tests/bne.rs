#[cfg(not(debug_assertions))]
use assert_approx_eq::assert_approx_eq;
use std::fs;
use tokenizers::models::bne::{BneTrainer, BNE};
use tokenizers::{
    normalizers, AddedToken, NormalizerWrapper, PreTokenizerWrapper, SplitDelimiterBehavior,
};
use tokenizers::{pre_tokenizers, DecoderWrapper, Model, Tokenizer};

#[test]
fn test_load_bne_from_file() {
    let model = BNE::from_file("data/bne-vocab.json", "data/bne-merges.txt")
        .build()
        .unwrap();
}

#[test]
fn test_decoding_with_added_bne() {
    let mut tokenizer = Tokenizer::from_file("data/bne.json").unwrap();
    tokenizer.with_pre_tokenizer(Some(PreTokenizerWrapper::from(
        pre_tokenizers::byte_level::ByteLevel::new(true, true, true),
    )));
    tokenizer.with_decoder(Some(DecoderWrapper::from(
        pre_tokenizers::byte_level::ByteLevel::new(true, true, true),
    )));
    let encoded = tokenizer
        .encode("Hey! how is this token: 嗎", false)
        .unwrap();
    assert_eq!(
        encoded.get_ids(),
        [11754, 5, 1853, 325, 558, 29106, 30, 11098, 198, 189]
    );
    assert_eq!(
        encoded.get_tokens(),
        ["ĠHey", "!", "Ġhow", "Ġis", "Ġthis", "Ġtoken", ":", "Ġå", "Ĺ", "İ"]
    );

    let decoded = tokenizer.decode(encoded.get_ids(), false);
    assert_eq!(decoded.unwrap(), " Hey! how is this token: 嗎");

    tokenizer.add_tokens(&[AddedToken::from("д", false).normalized(true)]);
    let encoded = tokenizer
        .encode("Hey! how is this token: д", false)
        .unwrap();
    assert_eq!(
        encoded.get_ids(),
        [11754, 5, 1853, 325, 558, 29106, 30, 174, 30000]
    );
    assert_eq!(
        encoded.get_tokens(),
        ["ĠHey", "!", "Ġhow", "Ġis", "Ġthis", "Ġtoken", ":", "Ġ", "д"]
    );
    let decoded = tokenizer.decode(encoded.get_ids(), false);
    assert_eq!(decoded.unwrap(), " Hey! how is this token: д")
}

#[test]
fn test_bne_model_from_file() {
    let model = BNE::from_file("data/BNE_32k-vocab.json", "data/BNE_32k-merges.txt")
        .build()
        .unwrap();
    let string = "Shall I compare thee to a summer’s day?";
    assert_eq!(
        model
            .tokenize(string)
            .unwrap()
            .iter()
            .map(|tok| tok.value.clone())
            .collect::<Vec<_>>(),
        vec!["Sh", "all", "I", "comp", "are", "the", "eto", "as", "umm", "ers", "day", "?"]
    );
}

#[test]
fn test_bne_tokenizer_from_file() {
    let mut tokenizer = Tokenizer::from_file("data/BNE_32k.json").unwrap();
    tokenizer.with_pre_tokenizer(Some(PreTokenizerWrapper::from(
        pre_tokenizers::byte_level::ByteLevel::new(true, true, true),
    )));
    tokenizer.with_decoder(Some(DecoderWrapper::from(
        pre_tokenizers::byte_level::ByteLevel::new(true, true, true),
    )));
    let string = fs::read_to_string("data/BNE_test_data_2.txt").unwrap();
    let encoded = tokenizer.encode(string, false).unwrap();
    assert_eq!(
        encoded.get_tokens(),
        vec![
            "ĠSh",
            "all",
            "ĠI",
            "Ġcompare",
            "Ġthe",
            "e",
            "Ġto",
            "Ġa",
            "Ġsummer",
            "âĢĻ",
            "s",
            "Ġday",
            "?",
            "Ċ",
            "Th",
            "ou",
            "Ġart",
            "Ġmore",
            "Ġlovely",
            "Ġand",
            "Ġmore",
            "Ġtemperate",
            ":",
            "Ċ",
            "R",
            "ough",
            "Ġwinds",
            "Ġdo",
            "Ġshake",
            "Ġthe",
            "Ġdar",
            "ling",
            "Ġbuds",
            "Ġof",
            "ĠMay",
            ",",
            "Ċ",
            "And",
            "Ġsummer",
            "âĢĻ",
            "s",
            "Ġlease",
            "Ġh",
            "ath",
            "Ġall",
            "Ġtoo",
            "Ġshort",
            "Ġa",
            "Ġdate",
            ";",
            "Ċ",
            "S",
            "omet",
            "ime",
            "Ġtoo",
            "Ġhot",
            "Ġthe",
            "Ġeye",
            "Ġof",
            "Ġheaven",
            "Ġshines",
            ",",
            "Ċ",
            "And",
            "Ġoften",
            "Ġis",
            "Ġhis",
            "Ġgold",
            "Ġcomplex",
            "ion",
            "Ġdim",
            "m",
            "'d",
            ";",
            "Ċ",
            "And",
            "Ġevery",
            "Ġfair",
            "Ġfrom",
            "Ġfair",
            "Ġsometime",
            "Ġdeclines",
            ",",
            "Ċ",
            "By",
            "Ġchance",
            "Ġor",
            "Ġnature",
            "âĢĻ",
            "s",
            "Ġchanging",
            "Ġcourse",
            "Ġun",
            "tr",
            "imm",
            "'d",
            ";",
            "Ċ",
            "But",
            "Ġth",
            "y",
            "Ġeternal",
            "Ġsummer",
            "Ġshall",
            "Ġnot",
            "Ġfade",
            ",",
            "Ċ",
            "N",
            "or",
            "Ġlose",
            "Ġpossession",
            "Ġof",
            "Ġthat",
            "Ġfair",
            "Ġth",
            "ou",
            "Ġow",
            "âĢĻ",
            "st",
            ";",
            "Ċ",
            "N",
            "or",
            "Ġshall",
            "Ġdeath",
            "Ġbr",
            "ag",
            "Ġth",
            "ou",
            "Ġwander",
            "âĢĻ",
            "st",
            "Ġin",
            "Ġhis",
            "Ġshade",
            ",",
            "Ċ",
            "When",
            "Ġin",
            "Ġeternal",
            "Ġlines",
            "Ġto",
            "Ġtime",
            "Ġth",
            "ou",
            "Ġgrow",
            "âĢĻ",
            "st",
            ":",
            "Ċ",
            "So",
            "Ġlong",
            "Ġas",
            "Ġmen",
            "Ġcan",
            "Ġbreathe",
            "Ġor",
            "Ġeyes",
            "Ġcan",
            "Ġsee",
            ",",
            "Ċ",
            "So",
            "Ġlong",
            "Ġlives",
            "Ġthis",
            ",",
            "Ġand",
            "Ġthis",
            "Ġgives",
            "Ġlife",
            "Ġto",
            "Ġthe",
            "e",
            ".",
            "Ċ"
        ]
    );
}

#[test]
fn test_bne_tokenizer_from_file_2() {
    let mut tokenizer = Tokenizer::from_file("data/BNE_32k.json").unwrap();
    tokenizer.with_pre_tokenizer(Some(PreTokenizerWrapper::from(
        pre_tokenizers::byte_level::ByteLevel::new(true, true, true),
    )));
    tokenizer.with_decoder(Some(DecoderWrapper::from(
        pre_tokenizers::byte_level::ByteLevel::new(true, true, true),
    )));
    let string = fs::read_to_string("data/BNE_test_data.txt").unwrap();
    let encoded = tokenizer.encode(string, false).unwrap();
    //assert_eq!(encoded.get_tokens(), vec!["ĠSh", "all", "ĠI", "Ġcompare", "Ġthe", "e", "Ġto", "Ġa", "Ġsummer", "âĢĻ", "s", "Ġday", "?", "Ċ", "Th", "ou", "Ġart", "Ġmore", "Ġlovely", "Ġand", "Ġmore", "Ġtemperate", ":", "Ċ", "R", "ough", "Ġwinds", "Ġdo", "Ġshake", "Ġthe", "Ġdar", "ling", "Ġbuds", "Ġof", "ĠMay", ",", "Ċ", "And", "Ġsummer", "âĢĻ", "s", "Ġlease", "Ġh", "ath", "Ġall", "Ġtoo", "Ġshort", "Ġa", "Ġdate", ";", "Ċ", "S", "omet", "ime", "Ġtoo", "Ġhot", "Ġthe", "Ġeye", "Ġof", "Ġheaven", "Ġshines", ",", "Ċ", "And", "Ġoften", "Ġis", "Ġhis", "Ġgold", "Ġcomplex", "ion", "Ġdim", "m", "'d", ";", "Ċ", "And", "Ġevery", "Ġfair", "Ġfrom", "Ġfair", "Ġsometime", "Ġdeclines", ",", "Ċ", "By", "Ġchance", "Ġor", "Ġnature", "âĢĻ", "s", "Ġchanging", "Ġcourse", "Ġun", "tr", "imm", "'d", ";", "Ċ", "But", "Ġth", "y", "Ġeternal", "Ġsummer", "Ġshall", "Ġnot", "Ġfade", ",", "Ċ", "N", "or", "Ġlose", "Ġpossession", "Ġof", "Ġthat", "Ġfair", "Ġth", "ou", "Ġow", "âĢĻ", "st", ";", "Ċ", "N", "or", "Ġshall", "Ġdeath", "Ġbr", "ag", "Ġth", "ou", "Ġwander", "âĢĻ", "st", "Ġin", "Ġhis", "Ġshade", ",", "Ċ", "When", "Ġin", "Ġeternal", "Ġlines", "Ġto", "Ġtime", "Ġth", "ou", "Ġgrow", "âĢĻ", "st", ":", "Ċ", "So", "Ġlong", "Ġas", "Ġmen", "Ġcan", "Ġbreathe", "Ġor", "Ġeyes", "Ġcan", "Ġsee", ",", "Ċ", "So", "Ġlong", "Ġlives", "Ġthis", ",", "Ġand", "Ġthis", "Ġgives", "Ġlife", "Ġto", "Ġthe", "e", ".", "Ċ"]);
}
