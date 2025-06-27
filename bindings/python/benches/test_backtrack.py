import os
import argparse
import datetime
from datasets import load_dataset
from tokenizers import Tokenizer
from typing import Tuple

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


def test(model: str, dataset: str, dataset_config: str):
    dataset_xnli = load_dataset(dataset, dataset_config)
    tokenizer = Tokenizer.from_pretrained(model)
    tokenizer2 = Tokenizer.from_pretrained(model)
    print("Enabling backtrack")
    tokenizer2.enable_backtrack()
    print("Enabled backtrack")

    for easy in [
        # "1880",
        # " cream",
        " Insectarium",
    ]:
        encoded = tokenizer.encode(easy)
        encoded2 = tokenizer2.encode(easy)
        if encoded.ids != encoded2.ids:
            import ipdb

            ipdb.set_trace()
        # assert encoded.ids == encoded2.ids

    sentences = []
    en_sentences = []
    for _i, item in enumerate(dataset_xnli["train"]):
        # sentence = item["premise"]["en"]
        # sentences.append(sentence)
        for lang, sentence in item["premise"].items():
            if lang == "en":
                en_sentences.append(sentence)
            sentences.append(sentence)
    sentences = en_sentences + sentences

    start = datetime.datetime.now()
    encoded = tokenizer.encode_batch_fast(sentences)
    print(f"Took {datetime.datetime.now() - start}")

    start = datetime.datetime.now()
    encoded2 = tokenizer2.encode_batch_fast(sentences)
    print(f"Took {datetime.datetime.now() - start}")

    assert len(encoded) == len(encoded2)
    assert len(encoded) == len(sentences)
    total = 0
    correct = 0
    smaller = 0
    larger = 0
    for enc, enc2, sentence in zip(encoded, encoded2, sentences):
        # if enc.ids != enc2.ids:
        #     print(enc.ids)
        #     print(enc2.ids)
        if enc.ids == enc2.ids:
            correct += 1
        if len(enc2.ids) < len(enc.ids):
            smaller += 1
        if len(enc.ids) < len(enc2.ids):
            larger += 1
        total += 1
        # assert enc.ids == enc2.ids, f"{enc.ids} != {enc2.ids} (Source: {sentence}"
    print(f"{correct} / {total} ({correct / total * 100:.2f}%%)")
    print(f"{smaller} / {smaller + larger} ({smaller / (smaller + larger) * 100:.2f}%%)")
    # print("All good !")


def main():
    parser = argparse.ArgumentParser(
        prog="bench_tokenizer",
        description="Getting a feel for speed when tokenizing",
    )
    parser.add_argument("-m", "--model", default=MODEL_ID, type=str)
    parser.add_argument("-d", "--dataset", default=DATASET, type=str)
    parser.add_argument("-ds", "--dataset-config", default=DATASET_CONFIG, type=str)
    args = parser.parse_args()
    test(args.model, args.dataset, args.dataset_config)


# Call the function to run the benchmark
if __name__ == "__main__":
    main()
