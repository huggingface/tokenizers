package co.hugginface.tokenizers;

import com.sun.jna.*;

public interface JnaCBertTokenizer extends Library {

    JnaCBertTokenizer INSTANCE = (JnaCBertTokenizer) Native.load("/Users/andreaduque/Workspace/tokenizers/bindings/java/src/main/tokenizers-jna/target/debug/libtokenizers_jna.dylib", JnaCBertTokenizer.class);

    Pointer CBertTokenizer_new();
    void CBertTokenizer_drop(Pointer tokenizer);
    void CBertTokenizer_some_method(Pointer tokenizer);
    void rust_function(String value);


}
