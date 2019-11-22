import co.huggingface.tokenizers.pretokenizers.WhitespacePretokenizer
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.Assertions.assertEquals


class TestWhitespacePretokenizer: NativeTest() {

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
}