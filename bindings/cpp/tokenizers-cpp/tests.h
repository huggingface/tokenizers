#pragma once

#define DOCTEST_CONFIG_IMPLEMENT

#include "tokenizers-cpp/tokenizer.h"
#include "tokenizers-cpp/tests.rs.h"
#include "rust/cxx.h"

#include <doctest/doctest.h>

#include <vector>
#include <string>
#include <memory>

namespace huggingface {
namespace tokenizers {
static std::string data_dir_;

inline std::string data_file(std::string name) {
    return data_dir_ + "/" + name;
}

inline std::string gpt2_vocab() { return data_file("gpt2-vocab.json"); }
inline std::string gpt2_merges() { return data_file("gpt2-merges.txt"); }
inline std::string bert_vocab() {
    return data_file("bert-base-uncased-vocab.txt");
}

#define COMPARE_CONTAINERS(actual, expected)                                  \
    do {                                                                      \
        auto actual_ = actual;                                                \
        auto expected_ = expected;                                            \
        REQUIRE(actual_.size() == expected_.size());                          \
        for (size_t i = 0; i < actual_.size(); i++) {                         \
            REQUIRE_MESSAGE(actual_[i] == expected_[i], "mismatch at " << i); \
        }                                                                     \
    } while (false)

TEST_SUITE("Normalizers") {
    void check_normalizer(const Normalizer& normalizer,
                          nonstd::string_view original,
                          nonstd::string_view expected) {
        NormalizedString n(original);
        normalizer.normalize(n);
        CHECK(n.get_normalized() == expected);
    }

    TEST_CASE("Replace literal") {
        // clang-format off
        /*
        let original = "This is a ''test''";
        let normalized = "This is a \"test\"";

        let mut n = NormalizedString::from(original);
        Replace::new("''", "\"").unwrap().normalize(&mut n).unwrap();

        assert_eq!(&n.get(), &normalized);
        */
        // clang-format on
        check_normalizer(Normalizer::replace_literal("''", "\""),
                         "This is a ''test''", "This is a \"test\"");
    }

    TEST_CASE("Replace regex") {
        // clang-format off
        /*
        let original = "This     is   a         test";
        let normalized = "This is a test";

        let mut n = NormalizedString::from(original);
        Replace::new(ReplacePattern::Regex(r"\s+".into()), ' ')
            .unwrap()
            .normalize(&mut n)
            .unwrap();

        assert_eq!(&n.get(), &normalized);
        */
        // clang-format on
        check_normalizer(Normalizer::replace_regex(R"(\s+)", " "),
                         "This     is   a         test", "This is a test");
    }

    TEST_CASE("NFKC") {
        // clang-format off
        /*
        let original = "\u{fb01}".to_string();
        let normalized = "fi".to_string();
        let mut n = NormalizedString::from(original.clone());
        NFKC.normalize(&mut n).unwrap();

        assert_eq!(
            n,
            NormalizedString::new(original, normalized, vec![(0, 3), (0, 3)], 0)
        );

        assert_eq!(n.alignments_original(), vec![(0, 2), (0, 2), (0, 2)]);
        */
        // clang-format on
        check_normalizer(Normalizer::nfkc(), "ﬁ", "fi");
    }

    TEST_CASE("Sequence") {
        // clang-format off
        /*
        let original: String = "Cụ thể, bạn sẽ tham gia một nhóm các giám đốc điều hành tổ chức, các nhà lãnh đạo doanh nghiệp, các học giả, chuyên gia phát triển và tình nguyện viên riêng biệt trong lĩnh vực phi lợi nhuận…".to_string();
        let normalized = "cu the, ban se tham gia mot nhom cac giam đoc đieu hanh to chuc, cac nha lanh đao doanh nghiep, cac hoc gia, chuyen gia phat trien va tinh nguyen vien rieng biet trong linh vuc phi loi nhuan...".to_string();
        let mut n = NormalizedString::from(original);
        NFKD.normalize(&mut n).unwrap();
        StripAccents.normalize(&mut n).unwrap();
        Lowercase.normalize(&mut n).unwrap();
        assert_eq!(&n.get(), &normalized);
        */
        // clang-format on
        std::string original =
            "Cụ thể, bạn sẽ tham gia một nhóm các giám đốc điều hành tổ chức, "
            "các nhà lãnh đạo doanh nghiệp, các học giả, chuyên gia phát triển "
            "và tình nguyện viên riêng biệt trong lĩnh vực phi lợi nhuận…";
        std::string expected =
            "cu the, ban se tham gia mot nhom cac giam đoc đieu hanh to chuc, "
            "cac nha lanh đao doanh nghiep, cac hoc gia, chuyen gia phat trien "
            "va tinh nguyen vien rieng biet trong linh vuc phi loi nhuan...";

        SUBCASE("dynamic") {
            std::vector<Normalizer> normalizers;
            normalizers.push_back(Normalizer::nfkd());
            normalizers.push_back(Normalizer::strip_accents());
            normalizers.push_back(Normalizer::lowercase());
            check_normalizer(Normalizer::sequence(normalizers), original,
                             expected);
        }

        SUBCASE("static") {
            check_normalizer(Normalizer::sequence(Normalizer::nfkd(),
                                                  Normalizer::strip_accents(),
                                                  Normalizer::lowercase()),
                             original, expected);
        }
    }
}

TEST_SUITE("Pre-tokenizers") {
    using StringAndOffsets = std::tuple<std::string, size_t, size_t>;

    void check_pre_tokenizer(const PreTokenizer& pre_tokenizer,
                             nonstd::string_view original,
                             const std::vector<StringAndOffsets>& expected) {
        PreTokenizedString pre_tokenized(original);
        pre_tokenizer.pre_tokenize(pre_tokenized);
        auto splits = pre_tokenized.get_splits(OffsetReferential::Original,
                                               OffsetType::Byte);
        CHECK(splits.size() == expected.size());
        for (int i = 0; i < splits.size(); i++) {
            StringAndOffsets actual{splits[i].original, splits[i].start,
                                    splits[i].end};
            CHECK_MESSAGE(actual == expected[i], "mismatched splits at " << i);
        }
    }

    TEST_CASE("BERT basic") {
        // clang-format off
        /*
        let pretok = BertPreTokenizer;
        let mut pretokenized: PreTokenizedString = "Hey friend!     How are you?!?".into();
        pretok.pre_tokenize(&mut pretokenized).unwrap();
        assert_eq!(
            pretokenized
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>(),
            vec![
                ("Hey", (0, 3)),
                ("friend", (4, 10)),
                ("!", (10, 11)),
                ("How", (16, 19)),
                ("are", (20, 23)),
                ("you", (24, 27)),
                ("?", (27, 28)),
                ("!", (28, 29)),
                ("?", (29, 30)),
            ]
        );
        */
        // clang-format on
        check_pre_tokenizer(PreTokenizer::bert(),
                            "Hey friend!     How are you?!?",
                            {{"Hey", 0, 3},
                             {"friend", 4, 10},
                             {"!", 10, 11},
                             {"How", 16, 19},
                             {"are", 20, 23},
                             {"you", 24, 27},
                             {"?", 27, 28},
                             {"!", 28, 29},
                             {"?", 29, 30}});
    }

    TEST_CASE("Byte-level") {
        // clang-format off
        /*
        let mut pretokenized = PreTokenizedString::from("Hello there\nHello there");
        let bytelevel = ByteLevel::default().add_prefix_space(false);
        bytelevel.pre_tokenize(&mut pretokenized).unwrap();

        assert_eq!(
            pretokenized
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>(),
            vec![
                ("Hello", (0, 5)),
                ("Ġthere", (5, 11)),
                ("Ċ", (11, 12)),
                ("Hello", (12, 17)),
                ("Ġthere", (17, 23))
            ]
        );
        */
        // clang-format on
        check_pre_tokenizer(PreTokenizer::byte_level(false),
                            "Hello there\nHello there",
                            {{"Hello", 0, 5},
                             {"Ġthere", 5, 11},
                             {"Ċ", 11, 12},
                             {"Hello", 12, 17},
                             {"Ġthere", 17, 23}});
    }

    TEST_CASE("Sequence") {
        // clang-format off
        /*
        let pretokenizers = vec![
            PreTokenizerWrapper::WhitespaceSplit(WhitespaceSplit),
            PreTokenizerWrapper::Punctuation(Punctuation),
        ];
        let pretok = Sequence::new(pretokenizers);
        let mut pretokenized: PreTokenizedString = "Hey friend!     How are you?!?".into();
        pretok.pre_tokenize(&mut pretokenized).unwrap();
        assert_eq!(
            pretokenized
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>(),
            vec![
                ("Hey", (0, 3)),
                ("friend", (4, 10)),
                ("!", (10, 11)),
                ("How", (16, 19)),
                ("are", (20, 23)),
                ("you", (24, 27)),
                ("?", (27, 28)),
                ("!", (28, 29)),
                ("?", (29, 30)),
            ]
        );
        */
        // clang-format on
        check_pre_tokenizer(
            PreTokenizer::sequence(PreTokenizer::whitespace_split(),
                                   PreTokenizer::punctuation()),
            "Hey friend!     How are you?!?",
            {{"Hey", 0, 3},
             {"friend", 4, 10},
             {"!", 10, 11},
             {"How", 16, 19},
             {"are", 20, 23},
             {"you", 24, 27},
             {"?", 27, 28},
             {"!", 28, 29},
             {"?", 29, 30}});
    }
}

TEST_SUITE("Models") {
    // these tests verify models can be initialized, actual use is in Tokenizers
    // tests
    TEST_CASE("BPE") {
        BpeBuilder().files(gpt2_vocab(), gpt2_merges()).build();

        BpeBuilder()
            .vocab_and_merges({{"a", 0}, {"b", 1}, {"ab", 2}}, {{"a", "b"}})
            .dropout(0.5f)
            .build();
    }

    TEST_CASE("WordPiece") {
        WordPieceBuilder().files(bert_vocab()).build();

        WordPieceBuilder()
            .vocab({{"a", 0}, {"b", 1}, {"ab", 2}})
            .continuing_subword_prefix("##")
            .build();
    }

    TEST_CASE("WordLevel") {
        WordLevelBuilder().files(gpt2_vocab()).build();

        WordLevelBuilder().vocab({{"a", 0}, {"b", 1}, {"ab", 2}}).build();

        std::unordered_map<std::string, uint32_t> vocab{
            {"a", 0}, {"b", 1}, {"ab", 2}};
        WordLevelBuilder().vocab(vocab).build();
    }

    TEST_CASE("Unigram") {
        Model::unigram(data_file("unigram.json"));

        Model::unigram({{"<unk>", 0.0}, {"hello", 1.0}}, 0);
    }
}

TEST_SUITE("PostProcessors") {
    TEST_CASE("Template") {
        // verifies BERT post-processor can be built with a
        // TemplateProcessingBuilder
        TemplateProcessingBuilder()
            .single("[CLS] $0 [SEP]")
            .pair("[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1")
            .special_tokens({{"[CLS]", 101}, {"[SEP]", 102}})
            .build();
    }
}

TEST_SUITE("Decoders") {
    TEST_CASE("Byte-level") {
        CHECK(Decoder::byte_level().decode({"My", "Ġname", "Ġis", "ĠJohn"}) ==
              "My name is John");
    }

    TEST_CASE("WordPiece") {
        CHECK(Decoder::word_piece().decode({"I", "'m", "Jo", "##hn"}) ==
              "I'm John");
        CHECK(Decoder::word_piece("__", false)
                  .decode({"I", "'m", "Jo", "__hn"}) == "I 'm John");
    }

    TEST_CASE("BPE") {
        CHECK(Decoder::bpe().decode({"My</w>", "na", "me</w>", "is</w>", "Jo",
                                     "hn</w>"}) == "My name is John");
    }

    TEST_CASE("Metaspace") {
        CHECK(Decoder::metaspace().decode({"▁My", "▁name", "▁is", "▁John"}) ==
              "My name is John");
        CHECK(Decoder::metaspace('-', false)
                  .decode({"-My", "-name", "-is", "-John"}) ==
              " My name is John");
    }
}

TEST_SUITE("Tokenizers") {
    void check_encoding(const Encoding& encoding,
                        const std::vector<rust::String>& expected_tokens,
                        const std::vector<uint32_t>& expected_ids = {}) {
        COMPARE_CONTAINERS(expected_tokens, encoding.get_tokens());
        if (!expected_tokens.empty() && !expected_ids.empty()) {
            COMPARE_CONTAINERS(expected_ids, encoding.get_ids());
        }
    }

    TEST_CASE("Bert") {
        Tokenizer tokenizer(WordPieceBuilder()
                                .files(bert_vocab())
                                .unk_token("[UNK]")
                                .max_input_chars_per_word(100)
                                .build());

        tokenizer.with_normalizer(Normalizer::bert())
            .with_pre_tokenizer(PreTokenizer::bert())
            .with_post_processor(
                PostProcessor::bert("[SEP]", 101, "[CLS]", 102));

        Encoding encoding = tokenizer.encode("My name is John", true);
        std::vector<rust::String> expected_tokens{"[CLS]", "my",   "name",
                                                  "is",    "john", "[SEP]"};
        COMPARE_CONTAINERS(expected_tokens, encoding.get_tokens());
        std::vector<uint32_t> expected_ids{102, 2026, 2171, 2003, 2198, 101};
        check_encoding(encoding, expected_tokens, expected_ids);

        tokenizer.with_padding({});
        std::vector<Encoding> batch_encoding =
            tokenizer.encode_batch({"My name is John", "My name"});
        check_encoding(batch_encoding[0], expected_tokens, expected_ids);
        check_encoding(batch_encoding[1],
                       {"[CLS]", "my", "name", "[SEP]", "[PAD]", "[PAD]"},
                       {102, 2026, 2171, 101, 0, 0});
    }

    TEST_CASE("GPT2") {
        Tokenizer tokenizer(
            BpeBuilder().files(gpt2_vocab(), gpt2_merges()).build());

        tokenizer.with_pre_tokenizer(PreTokenizer::byte_level(false));

        Encoding encoding = tokenizer.encode("My name is John", true);
        std::vector<rust::String> expected_tokens{"My", "Ġname", "Ġis",
                                                  "ĠJohn"};
        std::vector<uint32_t> expected_ids{3666, 1438, 318, 1757};
        check_encoding(encoding, expected_tokens, expected_ids);
    }
}

inline bool run_tests(rust::Str data_dir) {
    doctest::Context context;
    data_dir_ = std::string(data_dir);
    return context.run() == 0;
}
}  // namespace tokenizers
}  // namespace huggingface
