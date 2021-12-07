package co.huggingface.tokenizers;

import co.huggingface.tokenizers.ffi.FFIVec;
import com.sun.jna.Pointer;

/**
 * {@code Encodings} wraps a {@link co.huggingface.tokenizers.ffi.FFIVec} of {@link co.huggingface.tokenizers.Encodings}.
 * This is retrieved when batch encoding.
 */

public class Encodings {

    private Encoding[] encodings;

    /**
     * @param ptr points to a {@link co.huggingface.tokenizers.ffi.FFIVec} that contains the encodings.
     */
    public Encodings(Pointer ptr) {
        assert(ptr != null);

        var ffiVec = new FFIVec(ptr);
        int vecLen = ffiVec.len.intValue();
        this.encodings = new Encoding[vecLen];
        Pointer[] vecPointers = ffiVec.ptr.getPointerArray(0, vecLen);

        for (int i = 0; i < vecLen; i++) {
            encodings[i] = new Encoding(vecPointers[i]);
        }
    }

    public Encoding[] getEncodings(){
        return this.encodings;
    }
}
