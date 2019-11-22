package co.huggingface.tokenizers

import co.huggingface.tokenizers.jni.Native

class Token private constructor(val ref: Long): Native {

    fun tokenId(): Int {
        return nativelTokenId(ref);
    }

    fun value(): String{
        return nativeValue(ref);
    }

    fun offsetStart(): Int{
        return nativeOffsetStart(ref);
    }

    fun offsetEnd(): Int{
        return nativeOffsetEnd(ref)
    }

    override fun finalize(){
        nativeDestroy(ref)
    }

    // JNI methods bindings
    private external fun nativelTokenId(ref: Long): Int
    private external fun nativeValue(ref: Long): String
    private external fun nativeOffsetStart(ref: Long): Int
    private external fun nativeOffsetEnd(ref: Long): Int
    private external fun nativeDestroy(ref: Long): Void
}