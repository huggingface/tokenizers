from transformers import LlamaTokenizer, LlamaTokenizerFast
import time

tokenizer1 = LlamaTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf", split_special_tokens=True
)  # LlamaTokenizer
tokenizer2 = LlamaTokenizerFast.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf", split_special_tokens=True
)  # LlamaTokenizer
print(tokenizer1, tokenizer2)

s_time = time.time()
for i in range(1000):
    tokenizer1.tokenize("你好，where are you?" * 1000)
print(f"slow: {time.time() - s_time}")

s_time = time.time()
for i in range(1000):
    tokenizer2.tokenize("你好，where are you?" * 1000)
print(f"fast: {time.time() - s_time}")
