package co.huggingface.tokenizers.ffi;

import com.sun.jna.Pointer;
import com.sun.jna.Structure;

@Structure.FieldOrder({"ptr","len","cap"})
public class FFIVec extends Structure {
    public Pointer ptr;
    public FFILibrary.size_t len, cap;

    public FFIVec() {
    }

    public FFIVec(Pointer p) {
        super(p);
        read();
    }
}