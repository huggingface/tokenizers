package co.hugginface.tokenizers;

import com.sun.jna.*;

//Implement AutoCloseble?
class JTokenizer extends PointerType {
    private JnaJTokenizer INTERFACE = JnaJTokenizer.INSTANCE;

    //check if it isnt null and create exception if it is
    public JTokenizer(String identifier) {
        Pointer pointer = INTERFACE.JTokenizer_from_pretrained(identifier);
        this.setPointer(pointer);
    }
    public void close() {
        Pointer p = this.getPointer();
        INTERFACE.JTokenizer_drop(p);
    }
    public void printTokenizer(){
        Pointer p = this.getPointer();
        INTERFACE.JTokenizer_print_tokenizer(p);
    }

}

public class App {


    public static void main(String[] args) {
        JTokenizer tokenizer = new JTokenizer("xlm-roberta-base");
        tokenizer.printTokenizer();
        tokenizer.close();
    }
}