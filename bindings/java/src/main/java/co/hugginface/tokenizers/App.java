package co.hugginface.tokenizers;

public class App {


    public static void main(String[] args) {

        RustJNAInterface INTERFACE = RustJNAInterface.INSTANCE;

        INTERFACE.rust_function();


    }
}