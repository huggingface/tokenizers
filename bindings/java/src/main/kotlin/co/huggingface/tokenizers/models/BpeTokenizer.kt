package co.huggingface.tokenizers.models

class BpeTokenizer private constructor (val handle: Long){

    companion object {
        init {
            System.loadLibrary("tokenizers_jni")
        }

        fun fromFiles(vocabs: String, merges: String): BpeTokenizer {
            return BpeTokenizer(getHandle(vocabs, merges))
        }

        @JvmStatic private external fun getHandle(vocabs: String, merges: String): Long;
    }

    protected fun finalize() {

    }
}