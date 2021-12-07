package co.huggingface.tokenizers.ffi;

import com.sun.jna.*;

public interface FFILibrary extends Library {
    FFILibrary INSTANCE = Native.load("safer_ffi_tokenizers", FFILibrary.class);

    /**
     * {@code size_t} handles unsigned ints as size_t
     */
    public static class size_t extends IntegerType {
        public size_t() { this(0); }
        public size_t(long value) { super(Native.SIZE_T_SIZE, value); }
    }

    // getting & dropping tokenizers
    FFIResult tokenizer_from_pretrained(String identifier);
    void tokenizer_drop(Pointer tokenizerResult);

    // getting & dropping encoders
    FFIResult encode_from_str(Pointer tokenizer, String input, int addSpecialTokens);
    FFIResult encode_batch(Pointer tokenizer, FFIVec inputs, int addSpecialTokens);

    void encoding_drop(Pointer encodingResult);
    void encodings_drop(Pointer encodingsResult);
}