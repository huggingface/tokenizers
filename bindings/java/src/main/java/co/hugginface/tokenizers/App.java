package co.hugginface.tokenizers;

import com.sun.jna.*;

class CBertTokenizer extends PointerType {
    private JnaCBertTokenizer INTERFACE = JnaCBertTokenizer.INSTANCE;

    public CBertTokenizer() {
        Pointer pointer = INTERFACE.CBertTokenizer_new();
        this.setPointer(pointer);
    }
    public void close() {
        Pointer p = this.getPointer();
        INTERFACE.CBertTokenizer_drop(p);
    }
    public void someMethod(){
        Pointer p = this.getPointer();
        INTERFACE.CBertTokenizer_some_method(p);
    }

}

public class App {


    public static void main(String[] args) {
        CBertTokenizer tokenizer = new CBertTokenizer();
        tokenizer.someMethod();
        tokenizer.close();
    }
}