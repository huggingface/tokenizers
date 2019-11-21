import co.huggingface.tokenizers.Token
import co.huggingface.tokenizers.models.BpeTokenizer

fun main() {
//    val bpe = BpeTokenizer.fromFiles("D:\\Workspace\\Rust\\tokenizers\\data\\gpt2-vocab.json", "D:\\Workspace\\Rust\\tokenizers\\data\\gpt2-merges.txt")
//    print(bpe)

    val token = Token.getToken();
    println("TokenId: ${token.tokenId()}")
    println("Value: ${token.value()}")
    println("OffsetStart: ${token.offsetStart()}")
    println("OffsetEnd: ${token.offsetEnd()}")
}