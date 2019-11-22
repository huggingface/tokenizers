package co.huggingface.tokenizers.jni

/**
 * Interface for all the JNI allocated resources.
 * It provides all the method for managing resources' lifecycle.
 */
interface Native {

    /**
     * Called by the JVM when the object is about to be released by the GC.
     */
    fun finalize();
}