package co.hugginface.tokenizers;

import com.sun.jna.*;

import java.util.Arrays;
import java.util.List;

public class App {


    public static void main(String[] args) {
        String identifier = "bert-base-uncased";
        JnaJTokenizer.JTokenizer tokenizer = new JnaJTokenizer.JTokenizer(identifier);
        String tokenizeMe = "I love Java";
        List<Long> ids = tokenizer.encodeFromStr(tokenizeMe);

        System.out.println(String.format("ids from java: %s", Arrays.toString(ids.toArray())));
        tokenizer.close();
    }
}