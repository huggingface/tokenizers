package co.huggingface.tokenizers

data class Token constructor(val id: Long, val value: String, val offsetStart: Int, val offsetEnd: Int) {
}