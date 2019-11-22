package co.huggingface.tokenizers.pretokenizers

import co.huggingface.tokenizers.exceptions.NativeAllocationFailedException
import co.huggingface.tokenizers.exceptions.StringDecodingException

/**
 * A PreTokenizer takes care of pre-tokenizing strings before this goes to the model
 */
interface Pretokenizer {

    /**
     *
     */
    @Throws(NativeAllocationFailedException::class, StringDecodingException::class)
    fun pretokenize(s: String): Array<String>
}