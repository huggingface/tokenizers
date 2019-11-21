package co.huggingface.tokenizers

class Token private constructor(val ref: Long){

    companion object {
        init {
            System.loadLibrary("tokenizers_jni")
        }

        fun getToken(): Token{
            return Token(getHandle())
        }

        @JvmStatic private external fun getHandle(): Long
    }

    fun tokenId(): Int {
        return internalTokenId(ref);
    }

    fun value(): String{
        return internalValue(ref);
    }

    fun offsetStart(): Int{
        return internalOffsetStart(ref);
    }

    fun offsetEnd(): Int{
        return internalOffsetEnd(ref)
    }

    external fun internalTokenId(ref: Long): Int
    external fun internalValue(ref: Long): String
    external fun internalOffsetStart(ref: Long): Int
    external fun internalOffsetEnd(ref: Long): Int
}