#pragma once

#include "tokenizers-cpp/common.h"
#include "tokenizers-cpp/models.rs.h"

#include <nonstd/optional.hpp>

#include <string>
#include <unordered_map>

namespace huggingface {
namespace tokenizers {

struct BpeBuilder;
struct WordPieceBuilder;

struct Model {
    HFT_FFI_WRAPPER(Model);

public:
    static HFT_RESULT(Model) bpe(BpeBuilder& builder);
    static HFT_RESULT(Model) word_piece(WordPieceBuilder& builder);

    HFT_RESULT(rust::Vec<Token>) tokenize(nonstd::string_view sequence) {
        HFT_TRY(rust::Vec<Token>,
                ffi::tokenize(*inner_, string_view_to_str(sequence)));
    }

    nonstd::optional<uint32_t> token_to_id(nonstd::string_view token) {
        return HFT_OPTION(
            ffi::token_to_id_model(*inner_, string_view_to_str(token)));
    }

    nonstd::optional<std::string> id_to_token(uint32_t id) {
        ffi::OptionString opt_token(ffi::id_to_token_model(*inner_, id));
        return opt_token.has_value
                   ? nonstd::make_optional(std::string(opt_token.value))
                   : nonstd::nullopt;
    }

    size_t get_vocab_size() { return ffi::get_vocab_size_model(*inner_); }

    std::unordered_map<std::string, uint32_t> get_vocab() {
        rust::Vec<ffi::KVStringU32> entries(ffi::get_vocab_model(*inner_));
        std::unordered_map<std::string, uint32_t> vocab;
        for (auto& entry : entries) {
            vocab[std::string(entry.key)] = entry.value;
        }
        return vocab;
    }

    HFT_RESULT(rust::Vec<rust::String>) save(nonstd::string_view folder) {
        HFT_TRY(rust::Vec<rust::String>,
                ffi::save(*inner_, string_view_to_str(folder), false, {}));
    }

    HFT_RESULT(rust::Vec<rust::String>)
    save(nonstd::string_view folder, nonstd::string_view prefix) {
        HFT_TRY(rust::Vec<rust::String>,
                ffi::save(*inner_, string_view_to_str(folder), true,
                          string_view_to_str(prefix)));
    }
};

struct BpeBuilder {
    HFT_FFI_WRAPPER(BpeBuilder);

public:
    BpeBuilder() : inner_(ffi::bpe_builder()){};

    static BpeBuilder from_file(nonstd::string_view vocab,
                                nonstd::string_view merges) {
        BpeBuilder builder;
        builder.files(vocab, merges);
        return builder;
    }

    HFT_RESULT(Model) build() { HFT_TRY(Model, {inner_->build()}); }

    BpeBuilder& files(nonstd::string_view vocab, nonstd::string_view merges) {
        inner_->files(to_rust_string(vocab), to_rust_string(merges));
        return *this;
    }

    /// Vocab must be any container of std::pair<S, uint32_t>,
    /// Merges any container of std::pair<S, S>,
    /// where to_rust_string(S) returns a rust::String
    template <typename Vocab, typename Merges>
    BpeBuilder& vocab_and_merges(Vocab vocab, Merges merges) {
        rust::Vec<ffi::KVStringU32> vocab_ffi;
        fill_vec(vocab_ffi, vocab, [](auto& kv) {
            return {to_rust_string(kv.first), to_rust_string(kv.second)};
        });

        rust::Vec<ffi::StringString> merges_ffi;
        fill_vec(merges_ffi, merges, [](auto& kv) {
            return {to_rust_string(kv.first), to_rust_string(kv.second)};
        });

        inner_->vocab_and_merges(vocab_ffi, merges_ffi);
        return *this;
    }

    BpeBuilder& cache_capacity(size_t capacity) {
        inner_->cache_capacity(capacity);
        return *this;
    }

    BpeBuilder& unk_token(nonstd::string_view unk_token) {
        inner_->unk_token(to_rust_string(unk_token));
        return *this;
    }

    BpeBuilder& dropout(float dropout) {
        inner_->dropout(dropout);
        return *this;
    }

    BpeBuilder& continuing_subword_prefix(nonstd::string_view prefix) {
        inner_->continuing_subword_prefix(to_rust_string(prefix));
        return *this;
    }

    BpeBuilder& end_of_word_suffix(nonstd::string_view suffix) {
        inner_->end_of_word_suffix(to_rust_string(suffix));
        return *this;
    }

    BpeBuilder& fuse_unk(bool fuse_unk) {
        inner_->fuse_unk(fuse_unk);
        return *this;
    }
};

struct WordPieceBuilder {
    HFT_FFI_WRAPPER(WordPieceBuilder);

public:
    WordPieceBuilder() : inner_(ffi::word_piece_builder()){};

    static WordPieceBuilder from_file(nonstd::string_view vocab) {
        WordPieceBuilder builder;
        builder.files(vocab);
        return builder;
    }

    HFT_RESULT(Model) build() { HFT_TRY(Model, {inner_->build()}); }

    WordPieceBuilder& files(nonstd::string_view vocab) {
        inner_->files(string_view_to_str(vocab));
        return *this;
    }

    /// Vocab must be any container of std::pair<S, uint32_t>,
    /// where to_rust_string(S) returns a rust::String
    template <typename Vocab>
    WordPieceBuilder& vocab(Vocab vocab) {
        rust::Vec<ffi::KVStringU32> vocab_ffi;
        fill_vec(vocab_ffi, vocab, [](auto& kv) {
            return {to_rust_string(kv.first), to_rust_string(kv.second)};
        });

        inner_->vocab(vocab_ffi);
        return *this;
    }

    WordPieceBuilder& unk_token(nonstd::string_view unk_token) {
        inner_->unk_token(string_view_to_str(unk_token));
        return *this;
    }

    WordPieceBuilder& continuing_subword_prefix(nonstd::string_view prefix) {
        inner_->continuing_subword_prefix(string_view_to_str(prefix));
        return *this;
    }

    WordPieceBuilder& max_input_chars_per_word(
        size_t max_input_chars_per_word) {
        inner_->max_input_chars_per_word(max_input_chars_per_word);
        return *this;
    }
};

inline HFT_RESULT(Model) Model::bpe(BpeBuilder& builder) {
    return builder.build();
}

inline HFT_RESULT(Model) Model::word_piece(WordPieceBuilder& builder) {
    return builder.build();
}

}  // namespace tokenizers
}  // namespace huggingface
