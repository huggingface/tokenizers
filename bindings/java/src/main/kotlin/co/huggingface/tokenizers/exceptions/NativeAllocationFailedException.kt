package co.huggingface.tokenizers.exceptions

import java.lang.Exception

class NativeAllocationFailedException(val reason: String): Exception(reason) {
}