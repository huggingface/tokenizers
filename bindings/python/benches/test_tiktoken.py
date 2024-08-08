import os
import time
import argparse
from datasets import load_dataset
from tiktoken.load import load_tiktoken_bpe
import tiktoken
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download
from typing import Tuple, List
from multiprocessing import Process

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B"
DATASET = "facebook/xnli"
DATASET_CONFIG = "all_languages"
DEFAULT_THREADS = [2**i for i in range(8) if 2**i <= os.cpu_count()]


def format_byte_size(num_bytes: int) -> Tuple[str, str]:
    """Convert bytes to a human-readable format (KB, MB, GB)."""
    num_bytes_f = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes_f < 1024:
            return f"{num_bytes_f:.2f} {unit}", unit
        num_bytes_f /= 1024
    return f"{num_bytes_f:.2f} PB", "PB"


def benchmark_batch(model: str, documents: list[str], num_threads: int, document_length: float) -> None:
    os.environ["RAYON_NUM_THREADS"] = str(num_threads)
    num_bytes = sum(map(len, map(str.encode, documents)))
    readable_size, unit = format_byte_size(num_bytes)
    print(f"==============")
    print(f"num_threads: {num_threads}, data size: {readable_size}, documents: {len(documents)} Avg Length: {document_length:.0f}")
    filename = hf_hub_download(MODEL_ID, "original/tokenizer.model")
    mergeable_ranks = load_tiktoken_bpe(filename)
    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    num_reserved_special_tokens = 256
    special_tokens = [
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|reserved_special_token_0|>",
        "<|reserved_special_token_1|>",
        "<|reserved_special_token_2|>",
        "<|reserved_special_token_3|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|reserved_special_token_4|>",
        "<|eot_id|>",  # end of turn
    ] + [
        f"<|reserved_special_token_{i}|>"
        for i in range(5, num_reserved_special_tokens - 5)
    ]
    num_base_tokens = len(mergeable_ranks)
    special_tokens = {
        token: num_base_tokens + i for i, token in enumerate(special_tokens)
    }
    enc = tiktoken.Encoding(
            name=model,
            pat_str=pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )
    out = enc.encode("This is a test")

    hf_enc = Tokenizer.from_pretrained(model)
    out2 = hf_enc.encode("This is a test", add_special_tokens=False).ids

    assert out == out2, "sanity check"

    start = time.perf_counter_ns()
    enc.encode_ordinary_batch(documents, num_threads=num_threads)
    end = time.perf_counter_ns()

    readable_size, unit = format_byte_size(num_bytes / (end - start) * 1e9)
    print(f"tiktoken \t{readable_size}  / s")


    start = time.perf_counter_ns()
    hf_enc.encode_batch_fast(documents)
    end = time.perf_counter_ns()
    readable_size, unit = format_byte_size(num_bytes / (end - start) * 1e9)
    print(f"huggingface \t{readable_size} / s")


def test(model: str, dataset: str, dataset_config: str, threads: List[int]):
    dataset_xnli = load_dataset(dataset, dataset_config)

    input_lengths = [(10, False, True), (10_000, False, True), (10_000, False, False)]

    for num_threads in threads:
        for length, fuse, long in input_lengths:
            documents = []
            for i, item in enumerate(dataset_xnli["train"]):
                if i >= length:
                    break
                if long:
                    documents.append("".join(item["premise"].values()))
                else:
                    documents.append(item["premise"]["en"])
            if fuse:
                documents=["".join(documents)]

            document_length = sum(len(d) for d in documents) / len(documents)

            # Rayon thread pool is global to a process, we need to launch
            # separate processes in order to accurately use the correct number of threads.
            # Otherwise, we're simply running tokenizers in whatever tests comes first.
            # tokenizers does NOT provide a method to change the number of threads during
            # runtime.
            p = Process(target=benchmark_batch, args=(model, documents, num_threads, document_length))
            p.start()
            p.join()

            # benchmark_batch(model, documents, num_threads)


def main():

    parser = argparse.ArgumentParser(
                    prog='bench_tokenizer',
                    description='Getting a feel for speed when tokenizing',
    )
    parser.add_argument('-m', '--model', default=MODEL_ID, type=str)
    parser.add_argument('-d', '--dataset', default=DATASET, type=str)
    parser.add_argument('-ds', '--dataset-config', default=DATASET_CONFIG, type=str)
    parser.add_argument('-t', '--threads', nargs='+', default=DEFAULT_THREADS, type=int)
    args = parser.parse_args()
    test(args.model, args.dataset, args.dataset_config, args.threads)


# Call the function to run the benchmark
if __name__ == "__main__":
    main()
