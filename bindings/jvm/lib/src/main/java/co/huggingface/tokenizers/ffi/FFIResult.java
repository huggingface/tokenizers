package co.huggingface.tokenizers.ffi;

import com.sun.jna.Pointer;
import com.sun.jna.Structure;

@Structure.FieldOrder({"value","error"})
public class FFIResult extends Structure {
    public Pointer value;
    public String error;
}
