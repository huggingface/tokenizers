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
TEST_SUITE("Pre-tokenizers") {
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
        BertPreTokenizer pretok;
        PreTokenizedString pretokenized("Hey friend!     How are you?!?");
        pretok.pre_tokenize(pretokenized);
        auto splits = pretokenized.get_splits(OffsetReferential::Original,
                                              OffsetType::Byte);
        using StringAndOffsets = std::tuple<std::string, size_t, size_t>;
        std::vector<StringAndOffsets> expected{
            {"Hey", 0, 3},   {"friend", 4, 10}, {"!", 10, 11},
            {"How", 16, 19}, {"are", 20, 23},   {"you", 24, 27},
            {"?", 27, 28},   {"!", 28, 29},     {"?", 29, 30}};
        CHECK(splits.size() == expected.size());
        for (int i = 0; i < splits.size(); i++) {
            StringAndOffsets actual{splits[i].original, splits[i].start,
                                    splits[i].end};
            CHECK_MESSAGE(actual == expected[i], "mismatched splits at " << i);
        }
    }
}

TEST_SUITE("Tokenizers") {
    TEST_CASE("Example") {
        // example use of a tokenizer
        // just validates it can run and not throw
        Tokenizer tokenizer(BpeBuilder().build());

        tokenizer.with_normalizer(BertNormalizerOptions().build());

        tokenizer.encode("blablabla", true);
    }
}

inline bool run_tests() {
    doctest::Context context;
    return context.run() == 0;
}
}  // namespace tokenizers
}  // namespace huggingface
