package co.huggingface.tokenizers.pretokenizers

import co.huggingface.tokenizers.exceptions.NativeAllocationFailedException
import co.huggingface.tokenizers.exceptions.StringDecodingException
import co.huggingface.tokenizers.jni.Native

class WhitespacePretokenizer() : Pretokenizer, Native {

    private var ref: Long = -1

    init {
        this.ref = nativeHandle()
    }

    private external fun nativeHandle(): Long
    external override fun finalize();

    @Throws(NativeAllocationFailedException::class, StringDecodingException::class)
    external override fun pretokenize(s: String): Array<String>
}