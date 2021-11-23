package co.hugginface.tokenizers;

import com.sun.jna.*;

public interface JnaJTokenizer extends Library {

    JnaJTokenizer INSTANCE = (JnaJTokenizer) Native.load("/Users/andreaduque/Workspace/tokenizers/bindings/java/src/main/tokenizers-jna/target/debug/libtokenizers_jna.dylib", JnaJTokenizer.class);

    Pointer JTokenizer_from_pretrained(String identifier);
    void JTokenizer_drop(Pointer tokenizer);
    void JTokenizer_print_tokenizer(Pointer tokenizer);


}
