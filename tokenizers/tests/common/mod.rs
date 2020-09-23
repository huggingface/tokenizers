use tokenizers::decoders::wordpiece::WordPiece as WordPieceDecoder;
use tokenizers::models::bpe::BPE;
use tokenizers::models::wordpiece::WordPiece;
use tokenizers::normalizers::bert::BertNormalizer;
use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::processors::bert::BertProcessing;
use tokenizers::tokenizer::{Model, Tokenizer};

#[allow(dead_code)]
pub fn get_empty() -> Tokenizer {
    Tokenizer::new(BPE::default())
}

#[allow(dead_code)]
pub fn get_byte_level_bpe() -> BPE {
    BPE::from_file("data/gpt2-vocab.json", "data/gpt2-merges.txt")
        .build()
        .expect("Files not found, run `make test` to download these files")
}

#[allow(dead_code)]
pub fn get_byte_level(add_prefix_space: bool, trim_offsets: bool) -> Tokenizer {
    let mut tokenizer = Tokenizer::new(get_byte_level_bpe());
    tokenizer
        .with_pre_tokenizer(ByteLevel::default().add_prefix_space(add_prefix_space))
        .with_decoder(ByteLevel::default())
        .with_post_processor(ByteLevel::default().trim_offsets(trim_offsets));

    tokenizer
}

#[allow(dead_code)]
pub fn get_bert_wordpiece() -> WordPiece {
    WordPiece::from_file("data/bert-base-uncased-vocab.txt")
        .build()
        .expect("Files not found, run `make test` to download these files")
}

#[allow(dead_code)]
pub fn get_bert() -> Tokenizer {
    let mut tokenizer = Tokenizer::new(get_bert_wordpiece());
    let sep = tokenizer.get_model().token_to_id("[SEP]").unwrap();
    let cls = tokenizer.get_model().token_to_id("[CLS]").unwrap();
    tokenizer
        .with_normalizer(BertNormalizer::default())
        .with_pre_tokenizer(BertPreTokenizer)
        .with_decoder(WordPieceDecoder::default())
        .with_post_processor(BertProcessing::new(
            (String::from("[SEP]"), sep),
            (String::from("[CLS]"), cls),
        ));

    tokenizer
}
