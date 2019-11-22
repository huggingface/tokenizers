package co.huggingface.tokenizers.models

import co.huggingface.tokenizers.Token

/**
 * Represents a `Model` used during Tokenization (Like BPE or Word or Unigram)
 */
interface Model {


    /**
     * Tokenize the provided array of string to tokens.
     * @param words
     * @see co.huggingface.tokenizers.Token
     */
    fun tokenize(words: Array<String>): Array<Token>;
}