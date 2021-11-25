package co.hugginface.tokenizers;

import com.sun.jna.*;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public interface JnaJTokenizer extends Library {

    JnaJTokenizer INSTANCE = (JnaJTokenizer) Native.load("/Users/andreaduque/Workspace/tokenizers/bindings/java/src/main/tokenizers-jna/target/debug/libtokenizers_jna.dylib", JnaJTokenizer.class);

    //Implement AutoCloseble?
    class JTokenizer extends PointerType {

        //check if it isnt null and create exception if it is
        public JTokenizer(String identifier) {
            Pointer pointer = INSTANCE.JTokenizer_from_pretrained(identifier);
            this.setPointer(pointer);
        }
        public void close() {
            Pointer p = this.getPointer();
            INSTANCE.JTokenizer_drop(p);
        }
        public void printTokenizer(){
            Pointer p = this.getPointer();
            INSTANCE.JTokenizer_print_tokenizer(p);
        }
        //overloading with different types
        public List<Long> encodeFromStr(String value){
            Pointer p = this.getPointer();
            Pointer pEncodings = INSTANCE.JTokenizer_encode_from_str(p, value);
            JEncoding encoding = new JEncoding(pEncodings);
            List<Long> ids = encoding.getIds();
            encoding.close();
            return ids;
        }
    }


    //the encoding IDS are unsigned, but I think this isnt java supported

    public static class size_t extends IntegerType {
        public size_t() { this(0); }
        public size_t(long value) { super(Native.SIZE_T_SIZE, value); }
    }

    class JEncoding extends PointerType {

        public JEncoding(Pointer initializer) {
            this.setPointer(initializer);
        }
        public size_t getLength() {
            Pointer encodings = this.getPointer();
            size_t length = INSTANCE.JEncoding_get_length(encodings);
            return length;

        }
        //return Array of IDs
        public List<Long> getIds() {
            size_t idsSize = getLength();
            int isSizeInt = idsSize.intValue();
            Pointer buffer = new Memory((long) isSizeInt *Native.getNativeSize(long.class));
            Pointer encoding = this.getPointer();
            INSTANCE.JEncoding_get_ids(encoding, buffer, idsSize);
            long[] ids = buffer.getLongArray(0, isSizeInt);
            return Arrays.stream(ids).boxed().collect(Collectors.toList());
        }
        public void close() {
            Pointer p = this.getPointer();
            INSTANCE.JEncoding_drop(p);
        }
//        public void printTokenizer(){
//            Pointer p = this.getPointer();
//            INSTANCE.JTokenizer_print_tokenizer(p);
//        }
//        //overloading with different types
//        public void encodeFromStr(String value){
//            Pointer p = this.getPointer();
//            INSTANCE.JTokenizer_encode_from_str(p, value);
//        }
    }


    //give separate types for the different pointers
    //the way it is now is very error prone
    Pointer JTokenizer_from_pretrained(String identifier);
    void JTokenizer_drop(Pointer tokenizer);
    Pointer JTokenizer_encode_from_str(Pointer tokenizer, String input);
    void JTokenizer_print_tokenizer(Pointer tokenizer);
    void JEncoding_drop(Pointer tokenizer);
    size_t JEncoding_get_length(Pointer encoding);
    void JEncoding_get_ids(Pointer encoding, Pointer buffer, size_t sizeBuffer);


}
