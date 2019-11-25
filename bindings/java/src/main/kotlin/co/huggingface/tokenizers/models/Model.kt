package co.huggingface.tokenizers.models

import co.huggingface.tokenizers.Token
import java.util.*

/**
 * Represents a `Model` used during Tokenization (Like BPE or Word or Unigram)
 */
interface Model {


    /**
     * Tokenize the provided array of string to tokens.
     * @param words
     * @see co.huggingface.tokenizers.Token
     */
    fun tokenize(words: List<String>): List<Token>;
    fun decode(ids: List<Int>): List<String>;
    fun token_to_id(token: String): Optional<Int>;
    fun id_to_token(id: Int): Optional<String>;
}