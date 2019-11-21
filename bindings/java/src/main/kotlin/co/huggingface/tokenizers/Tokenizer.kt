package co.huggingface.tokenizers

class Tokenizer(val ref: Long) {
    external fun encode(text: String): Array<String>
}