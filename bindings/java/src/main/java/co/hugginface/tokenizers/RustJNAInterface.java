package co.hugginface.tokenizers;

import com.sun.jna.Library;
import com.sun.jna.Native;

public interface RustJNAInterface extends Library {

    RustJNAInterface INSTANCE = (RustJNAInterface) Native.load("/Users/andreaduque/Workspace/tokenizers/bindings/java/src/main/tokenizers-jna/target/debug/libtokenizers_jna.dylib", RustJNAInterface.class);

    void rust_function();

}
