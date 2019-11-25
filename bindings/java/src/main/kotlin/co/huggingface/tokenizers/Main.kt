package co.huggingface.tokenizers

import co.huggingface.tokenizers.Token
import co.huggingface.tokenizers.models.BytePairEncoder
import co.huggingface.tokenizers.pretokenizers.WhitespacePretokenizer

object Main {
    @JvmStatic
    fun main(args: Array<String>) {
        System.loadLibrary("tokenizers_jni")
        val t = BytePairEncoder.fromFiles("D:\\Workspace\\Rust\\tokenizers\\data\\gpt2-vocab.json", "D:\\Workspace\\Rust\\tokenizers\\data\\gpt2-merges.txt")
        println(t.handle)

        println(t.tokenize("My name is Morgan".split(" ")))
    }
}