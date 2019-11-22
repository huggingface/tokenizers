import co.huggingface.tokenizers.pretokenizers.WhitespacePretokenizer
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNotEquals


class TestWhitespacePretokenizer: NativeTest() {

    @Test
    fun testInit(){
        val tokenizer = WhitespacePretokenizer()
        val field = tokenizer.javaClass.getDeclaredField("ref")
        field.trySetAccessible()
        assertNotEquals(field.get(tokenizer), -1)
    }

    @Test
    fun testPretokenizeEmptyString(){
        val tokenizer = WhitespacePretokenizer()
        val tokens = tokenizer.pretokenize("")
        assertEquals(tokens.size, 0)
    }

    @Test
    fun testPretokenizeString(){
        val STR = "I work at HuggingFace"
        val tokenizer = WhitespacePretokenizer()
        val tokens = tokenizer.pretokenize(STR)
        assertEquals(tokens.size, 4)
        assertEquals(tokens, STR.split(" "))
    }

    @Test
    fun testFinalize(){
        val tokenizer = WhitespacePretokenizer()
        tokenizer.finalize()
        val field = tokenizer.javaClass.getDeclaredField("ref")
        field.trySetAccessible()
        assertEquals(field.get(tokenizer), -1L)
    }
}