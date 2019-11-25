package co.huggingface.tokenizers.models

import co.huggingface.tokenizers.Token
import co.huggingface.tokenizers.jni.Native
import java.util.*

class BytePairEncoder private constructor(): Model, Native {

    public var handle: Long = -1

    companion object{
        @JvmStatic
        external fun fromFiles(vocabs: String, merges: String): BytePairEncoder
    }

    external override fun tokenize(words: List<String>): List<Token>
    external override fun decode(ids: List<Int>): List<String>
    external override fun token_to_id(token: String): Optional<Int>
    external override fun id_to_token(id: Int): Optional<String>
    external override fun finalize()
}