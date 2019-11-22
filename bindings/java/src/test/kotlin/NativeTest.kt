import org.junit.jupiter.api.AfterAll
import org.junit.jupiter.api.BeforeAll

abstract class NativeTest {
    companion object{
        @BeforeAll
        @JvmStatic
        fun initialize(){
            System.loadLibrary("tokenizers_jni");
        }
    }
}