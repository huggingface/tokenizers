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
struct WordLevelBuilder;

/**
 * @brief Represents a model used during Tokenization (like BPE or Word or
 * Unigram).
 */
struct Model {
    HFT_FFI_WRAPPER(Model);

public:
    /**
     * @brief Builds a BPE model with given options.
     *
     * @see BpeBuilder
     */
    static HFT_RESULT(Model) bpe(BpeBuilder& builder);

    /**
     * @brief Builds a WordPiece model with given options.
     *
     * @see WordPieceBuilder
     */
    static HFT_RESULT(Model) word_piece(WordPieceBuilder& builder);

    /**
     * @brief Builds a word-level model with given options.
     *
     * @see WordLevelBuilder
     */
    static HFT_RESULT(Model) word_level(WordLevelBuilder& builder);

    /**
     * @brief Builds an unigram model.
     *
     * @tparam Vocab A container of `std::pair<S, double>`, where S can be
     * converted to rust::String
     * @param vocab Pairs of tokens and their logprobs.
     * @param unk_id Id of the unknown token.
     */
    template <typename Vocab>
    static HFT_RESULT(Model)
        unigram(Vocab vocab, nonstd::optional<size_t> unk_id) {
        HFT_TRY(Model, [&]() {
            rust::Vec<UnigramEntry> vocab_ffi;
            ffi::fill_vec(vocab_ffi, vocab, [](auto& pair) {
                return UnigramEntry{ffi::to_rust_string(pair.first),
                                    pair.second};
            });
            return Model(ffi::unigram_model(ffi::to_rust_slice(vocab_ffi),
                                            unk_id.has_value(),
                                            unk_id.value_or(0)));
        }());
    }

    /**
     * @brief Builds an unigram model from list-initialized vocab.
     *
     * @param vocab Pairs of tokens and their logprobs.
     * @param unk_id Id of the unknown token.
     */
    static HFT_RESULT(Model) unigram(std::initializer_list<UnigramEntry> vocab,
                                     nonstd::optional<size_t> unk_id) {
        HFT_TRY(Model, Model(ffi::unigram_model(ffi::to_rust_slice(vocab),
                                                unk_id.has_value(),
                                                unk_id.value_or(0))));
    }

    /**
     * @brief Loads an unigram model from a file.
     *
     * @param path Path to JSON file containing the serialized model.
     */
    static HFT_RESULT(Model) unigram(nonstd::string_view path) {
        HFT_TRY(Model, {ffi::unigram_load_model(ffi::to_rust_str(path))});
    }

    /**
     * @brief Tokenize the given sequence into multiple underlying Token s. The
     * offsets on the Token s are expected to be relative to the given
     * sequence.
     */
    HFT_RESULT(rust::Vec<Token>) tokenize(nonstd::string_view sequence) {
        HFT_TRY(rust::Vec<Token>,
                ffi::tokenize(*inner_, ffi::to_rust_str(sequence)));
    }

    /**
     * @brief Find the ID associated to a string token.
     */
    nonstd::optional<uint32_t> token_to_id(nonstd::string_view token) {
        return ffi::to_optional(
            ffi::token_to_id_model(*inner_, ffi::to_rust_str(token)));
    }

    /**
     * @brief Find the string token associated to an ID.
     */
    nonstd::optional<std::string> id_to_token(uint32_t id) {
        ffi::OptionString opt_token(ffi::id_to_token_model(*inner_, id));
        return opt_token.has_value
                   ? nonstd::make_optional(std::string(opt_token.value))
                   : nonstd::nullopt;
    }

    /**
     * @brief Returns the vocabulary size.
     */
    size_t get_vocab_size() { return ffi::get_vocab_size_model(*inner_); }

    /**
     * @brief Returns the entire vocabulary mapping (token -> ID).
     */
    std::unordered_map<std::string, uint32_t> get_vocab() {
        rust::Vec<TokenAndId> entries(ffi::get_vocab_model(*inner_));
        std::unordered_map<std::string, uint32_t> vocab;
        for (auto& entry : entries) {
            vocab[std::string(entry.token)] = entry.id;
        }
        return vocab;
    }

    /**
     * @brief Saves the Model.
     *
     * @param folder the folder to save the Model in.
     */
    HFT_RESULT(rust::Vec<rust::String>) save(nonstd::string_view folder) {
        HFT_TRY(rust::Vec<rust::String>,
                ffi::save(*inner_, ffi::to_rust_str(folder), false, {}));
    }

    /**
     * @brief Saves the Model.
     *
     * @param folder the folder to save the Model in.
     * @param prefix the prefix to use for file names.
     */
    HFT_RESULT(rust::Vec<rust::String>)
    save(nonstd::string_view folder, nonstd::string_view prefix) {
        HFT_TRY(rust::Vec<rust::String>,
                ffi::save(*inner_, ffi::to_rust_str(folder), true,
                          ffi::to_rust_str(prefix)));
    }
};

/**
 * @brief BPE model builder.
 */
struct BpeBuilder {
    HFT_FFI_WRAPPER(BpeBuilder);

public:
    /**
     * @brief The default BpeBuilder.
     */
    BpeBuilder() : inner_(ffi::bpe_builder()){};

    /**
     * @brief Creates the BpeBuilder with the given file names.
     *
     * Equivalent to `BpeBuilder().files(vocab, merges)`.
     *
     * @param vocab The vocab file name.
     * @param merges The merges file name.
     */
    static BpeBuilder from_file(nonstd::string_view vocab,
                                nonstd::string_view merges) {
        BpeBuilder builder;
        builder.files(vocab, merges);
        return builder;
    }

    /**
     * @brief Builds the model.
     */
    HFT_RESULT(Model) build() { HFT_TRY(Model, {inner_->build()}); }

    /**
     * @brief Sets the file names.
     *
     * @param vocab The vocab file name.
     * @param merges The merges file name.
     */
    BpeBuilder& files(nonstd::string_view vocab, nonstd::string_view merges) {
        inner_->files(ffi::to_rust_str(vocab), ffi::to_rust_str(merges));
        return *this;
    }

    /**
     * @brief Sets vocab and merges.
     *
     * @tparam Vocab A container of std::pair<S, uint32_t>, where S can be
     * converted to rust::String
     * @tparam Merges A container of std::pair<S, S>
     * @param vocab The vocabulary
     * @param merges The merges
     */
    template <typename Vocab, typename Merges>
    BpeBuilder& vocab_and_merges(Vocab vocab, Merges merges) {
        rust::Vec<TokenAndId> vocab_ffi;
        fill_vec(vocab_ffi, vocab, [](auto& kv) {
            return TokenAndId{ffi::to_rust_string(kv.first), kv.second};
        });

        rust::Vec<Merge> merges_ffi;
        fill_vec(merges_ffi, merges, [](auto& kv) {
            return Merge{ffi::to_rust_string(kv.first),
                         ffi::to_rust_string(kv.second)};
        });

        inner_->vocab_and_merges(ffi::to_rust_slice(vocab_ffi),
                                 ffi::to_rust_slice(merges_ffi));
        return *this;
    }

    /**
     * @brief Sets vocab and merges using list-initialization.
     *
     * @param vocab The vocabulary
     * @param merges The merges
     */
    BpeBuilder& vocab_and_merges(std::initializer_list<TokenAndId> vocab,
                                 std::initializer_list<Merge> merges) {
        inner_->vocab_and_merges(ffi::to_rust_slice(vocab),
                                 ffi::to_rust_slice(merges));
        return *this;
    }

    /**
     * @brief Sets the cache capacity. Set to 0 if you want to disable caching.
     */
    BpeBuilder& cache_capacity(size_t capacity) {
        inner_->cache_capacity(capacity);
        return *this;
    }

    /**
     * @brief Sets the unknown token.
     */
    BpeBuilder& unk_token(nonstd::string_view unk_token) {
        inner_->unk_token(ffi::to_rust_str(unk_token));
        return *this;
    }

    /**
     * @brief Sets the dropout value (0.0 for no dropout, 1.0 for no merges).
     */
    BpeBuilder& dropout(float dropout) {
        inner_->dropout(dropout);
        return *this;
    }

    /**
     * @brief Sets the continuing subword prefix.
     */
    BpeBuilder& continuing_subword_prefix(nonstd::string_view prefix) {
        inner_->continuing_subword_prefix(ffi::to_rust_str(prefix));
        return *this;
    }

    /**
     * @brief Sets the end-of-word suffix.
     */
    BpeBuilder& end_of_word_suffix(nonstd::string_view suffix) {
        inner_->end_of_word_suffix(ffi::to_rust_str(suffix));
        return *this;
    }

    /**
     * @brief Sets whether to fuse unknown token.
     */
    BpeBuilder& fuse_unk(bool fuse_unk) {
        inner_->fuse_unk(fuse_unk);
        return *this;
    }
};

/**
 * @brief WordPiece model builder.
 */
struct WordPieceBuilder {
    HFT_FFI_WRAPPER(WordPieceBuilder);

public:
    /**
     * @brief The default WordPieceBuilder.
     */
    WordPieceBuilder() : inner_(ffi::word_piece_builder()){};

    /**
     * @brief Creates the WordPieceBuilder with the given file name.
     *
     * Equivalent to `WordPieceBuilder().files(vocab)`.
     *
     * @param vocab The vocab file name.
     */
    static WordPieceBuilder from_file(nonstd::string_view vocab) {
        WordPieceBuilder builder;
        builder.files(vocab);
        return builder;
    }

    /**
     * @brief Builds the model.
     */
    HFT_RESULT(Model) build() { HFT_TRY(Model, {inner_->build()}); }

    /**
     * @brief Sets the file names.
     *
     * @param vocab The vocab file name.
     */
    WordPieceBuilder& files(nonstd::string_view vocab) {
        inner_->files(ffi::to_rust_str(vocab));
        return *this;
    }

    /**
     * @brief Sets vocab.
     *
     * @tparam Vocab A container of std::pair<S, uint32_t>, where S can be
     * converted to rust::String
     * @param vocab The vocabulary
     */
    template <typename Vocab>
    WordPieceBuilder& vocab(Vocab vocab) {
        rust::Vec<TokenAndId> vocab_ffi;
        fill_vec(vocab_ffi, vocab, [](auto& kv) {
            return TokenAndId{ffi::to_rust_string(kv.first), kv.second};
        });

        inner_->vocab(ffi::to_rust_slice(vocab_ffi));
        return *this;
    }

    /**
     * @brief Sets list-initialized vocab.
     *
     * @param vocab The vocabulary
     */
    WordPieceBuilder& vocab(std::initializer_list<TokenAndId> vocab) {
        inner_->vocab(ffi::to_rust_slice(vocab));
        return *this;
    }

    /**
     * @brief Sets the unknown token.
     */
    WordPieceBuilder& unk_token(nonstd::string_view unk_token) {
        inner_->unk_token(ffi::to_rust_str(unk_token));
        return *this;
    }

    /**
     * @brief Sets the continuing subword prefix.
     */
    WordPieceBuilder& continuing_subword_prefix(nonstd::string_view prefix) {
        inner_->continuing_subword_prefix(ffi::to_rust_str(prefix));
        return *this;
    }

    /**
     * @brief Sets the maximum number of characters per word.
     *
     * Any larger words will automatically be considered unknown.
     */
    WordPieceBuilder& max_input_chars_per_word(
        size_t max_input_chars_per_word) {
        inner_->max_input_chars_per_word(max_input_chars_per_word);
        return *this;
    }
};

/**
 * @brief Word-level model builder.
 *
 * A word-level model simply looks up each part of PreTokenizedString in its
 * vocabulary.
 */
struct WordLevelBuilder {
    HFT_FFI_WRAPPER(WordLevelBuilder);

public:
    /**
     * @brief The default WordLevelBuilder.
     */
    WordLevelBuilder() : inner_(ffi::word_level_builder()){};

    /**
     * @brief Creates the WordLevelBuilder with the given file name.
     *
     * Equivalent to `WordLevelBuilder().files(vocab)`.
     *
     * @param vocab The vocab file name.
     */
    static WordLevelBuilder from_file(nonstd::string_view vocab) {
        WordLevelBuilder builder;
        builder.files(vocab);
        return builder;
    }

    /**
     * @brief Builds the model.
     */
    HFT_RESULT(Model) build() { HFT_TRY(Model, {inner_->build()}); }

    /**
     * @brief Sets the file names.
     *
     * @param vocab The vocab file name.
     */
    WordLevelBuilder& files(nonstd::string_view vocab) {
        inner_->files(ffi::to_rust_str(vocab));
        return *this;
    }

    /**
     * @brief Sets vocab.
     *
     * @tparam Vocab A container of std::pair<S, uint32_t>, where S can be
     * converted to rust::String
     * @param vocab The vocabulary
     */
    template <typename Vocab>
    WordLevelBuilder& vocab(Vocab vocab) {
        rust::Vec<TokenAndId> vocab_ffi;
        ffi::fill_vec(vocab_ffi, vocab, [](auto& kv) {
            return TokenAndId{ffi::to_rust_string(kv.first), kv.second};
        });

        inner_->vocab(ffi::to_rust_slice(vocab_ffi));
        return *this;
    }

    /**
     * @brief Sets list-initialized vocab.
     *
     * @param vocab The vocabulary
     */
    WordLevelBuilder& vocab(std::initializer_list<TokenAndId> vocab) {
        inner_->vocab(ffi::to_rust_slice(vocab));
        return *this;
    }

    /**
     * @brief Sets the unknown token.
     */
    WordLevelBuilder& unk_token(nonstd::string_view unk_token) {
        inner_->unk_token(ffi::to_rust_str(unk_token));
        return *this;
    }
};

inline HFT_RESULT(Model) Model::bpe(BpeBuilder& builder) {
    return builder.build();
}

inline HFT_RESULT(Model) Model::word_level(WordLevelBuilder& builder) {
    return builder.build();
}

}  // namespace tokenizers
}  // namespace huggingface
