package co.huggingface.tokenizers;

import co.huggingface.tokenizers.ffi.FFILibrary;
import co.huggingface.tokenizers.ffi.FFIResult;
import co.huggingface.tokenizers.ffi.FFIVec;
import com.sun.jna.*;

import java.lang.ref.Cleaner;
import java.util.List;

/**
 *  {@code TokenizerFromPretrained} loads a pretrained tokenizer from the Hub given a identifier.
 *  It will wrap ffi calls in {@link co.huggingface.tokenizers.Result}
 */
public class TokenizerFromPretrained {
    private Pointer pointer;

    // according to https://techinplanet.com/java-9-cleaner-cleaner-cleanable-objects/,
    // it is wise to keep the cleaner runnables as a static class
    // to automatically free memory on the Rust side when GC'ed on JVM
    private static final Cleaner cleaner = Cleaner.create();
    private static class CleanTokenizer implements Runnable {
        private FFIResult result;

        public CleanTokenizer(FFIResult result) {
            this.result = result;
        }

        @Override
        public void run() {
            FFILibrary.INSTANCE.tokenizer_drop(result.getPointer());
        }
    }

    private TokenizerFromPretrained(Pointer pointer) {
        assert(pointer != null);
        this.pointer = pointer;
    }

    /**
     * @param identifier model indentifier from the hub
     * @return A successful tokenizer instance or an error
     */
    public static Result<TokenizerFromPretrained> create(String identifier) {
        var ffiResult = FFILibrary.INSTANCE.tokenizer_from_pretrained(identifier);
        TokenizerFromPretrained wrapper = null;

        if (ffiResult.value != null) {
            wrapper = new TokenizerFromPretrained(ffiResult.value);
            cleaner.register(wrapper, new CleanTokenizer(ffiResult));
        } else {
            FFILibrary.INSTANCE.tokenizer_drop(ffiResult.getPointer());
        }

        return new Result<TokenizerFromPretrained>(ffiResult, wrapper);
    }

    /**
     * @param input the text input as String to encode
     * @param addSpecialTokens Boolean to add special tokens or not
     * @return the encoding results
     */
    public Result<Encoding> encode(String input, Boolean addSpecialTokens) {
        var ffiResult = FFILibrary.INSTANCE.encode_from_str(this.pointer, input, addSpecialTokens ? 1 : 0);
        var wrapper = ffiResult.value == null ? null : new Encoding(ffiResult.value);
        FFILibrary.INSTANCE.encoding_drop(ffiResult.getPointer());

        return new Result<Encoding>(ffiResult, wrapper);
    }

    /**
     * It should run encoding in parallel for each item on the list if  TOKENIZERS_PARALLELISM=true
     *
     * @param input a batch of text inputs as a {@link
         java.util.List} of Strings
     * @param addSpecialTokens Boolean to add special tokens or not
     * @return the encoding results
     */

    public Result<Encodings> encode_batch(List<String> input, Boolean addSpecialTokens) {
        FFIVec vec = new FFIVec();
        vec.ptr = new StringArray(input.toArray(new String[0]));
        vec.len = new FFILibrary.size_t(input.size());
        vec.cap = new FFILibrary.size_t(input.size());
        var ffiResult =  FFILibrary.INSTANCE.encode_batch(this.pointer, vec, addSpecialTokens ? 1 : 0);
        var wrapper = ffiResult.value == null ? null : new Encodings(ffiResult.value);
        FFILibrary.INSTANCE.encodings_drop(ffiResult.getPointer());

        return new Result<Encodings>(ffiResult, wrapper);
    }
}
