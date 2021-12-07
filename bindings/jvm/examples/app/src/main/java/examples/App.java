package examples;
import  co.huggingface.tokenizers.*;
import java.util.ArrayList;
import java.util.Arrays;


public class App {

    public static void main(String[] args) {
        var str = "My name is John";

        var tk1 = TokenizerFromPretrained.create("bert-base-caseaoeuoaeuoaueoaud");
        System.err.println(tk1.error());

        var tk2 = TokenizerFromPretrained.create("bert-base-cased");
        var tokenizer = tk2.value();

        var encoding = tokenizer.encode(str, true);
        System.out.println(encoding.value());
        System.out.println(Arrays.toString(encoding.value().getTokens()));
        var list = new ArrayList<String>();
        list.add("Hello world");
        list.add("I love Java");
        list.add("My name is Viet and Andrea");
        var encodings = tokenizer.encode_batch(list, true).value();
        System.out.println(Arrays.toString(encodings.getEncodings()[2].getTokens()));
    }
}
