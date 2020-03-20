use tokenizers::decoders::wordpiece::WordPiece as WordPieceDecoder;
use tokenizers::models::bpe::BPE;
use tokenizers::models::wordpiece::WordPiece;
use tokenizers::normalizers::bert::BertNormalizer;
use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::processors::bert::BertProcessing;
use tokenizers::tokenizer::Tokenizer;

pub fn get_byte_level(add_prefix_space: bool, trim_offsets: bool) -> Tokenizer {
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

pub fn get_bert() -> Tokenizer {
    let mut tokenizer = Tokenizer::new(Box::new(
        WordPiece::from_files("data/bert-base-uncased-vocab.txt")
            .build()
            .expect("Files not found, run `make test` to download these files"),
    ));
    tokenizer.with_normalizer(Box::new(BertNormalizer::default()));
    tokenizer.with_pre_tokenizer(Box::new(BertPreTokenizer));
    tokenizer.with_decoder(Box::new(WordPieceDecoder::default()));
    tokenizer.with_post_processor(Box::new(BertProcessing::new(
        (
            String::from("[SEP]"),
            tokenizer.get_model().token_to_id("[SEP]").unwrap(),
        ),
        (
            String::from("[CLS]"),
            tokenizer.get_model().token_to_id("[CLS]").unwrap(),
        ),
    )));

    tokenizer
}
