import argparse
import glob
import json
import os
from os.path import join

from tokenizers import Tokenizer, normalizers, trainers
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import RandomChunkSplit


parser = argparse.ArgumentParser()
parser.add_argument(
    "--files",
    default=None,
    metavar="path",
    type=str,
    required=True,
    help="The files to use as training; accept '**/*.txt' type of patterns \
                          if enclosed in quotes",
)
parser.add_argument(
    "--out",
    default="./",
    type=str,
    help="Path to the output directory, where the files will be saved",
)
parser.add_argument("--name", default="random-chunk-bpe", type=str, help="The name of the output vocab files")
parser.add_argument("--min-length", default=2, type=int, help="Minimum length of chunks")
parser.add_argument("--max-length", default=5, type=int, help="Maximum length of chunks")
parser.add_argument("--vocab-size", default=10000, type=int, help="Size of vocabulary")
parser.add_argument("--min-frequency", default=2, type=int, help="Minimum frequency for a token to be included")
args = parser.parse_args()

files = glob.glob(args.files)
if not files:
    print(f"File does not exist: {args.files}")
    exit(1)


# Initialize a tokenizer with BPE model
tokenizer = Tokenizer(BPE())

# Use RandomChunkSplit as pre-tokenizer
tokenizer.pre_tokenizer = RandomChunkSplit(min_length=args.min_length, max_length=args.max_length)

# Optional: Add NFKC normalization like SentencePieceBPE
tokenizer.normalizer = normalizers.NFKC()

# Configure the BPE trainer
trainer = trainers.BpeTrainer(
    vocab_size=args.vocab_size,
    min_frequency=args.min_frequency,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    show_progress=True
)

# Train the model
print(f"Training BPE with RandomChunkSplit (min_length={args.min_length}, max_length={args.max_length})")
tokenizer.train(files, trainer)

# Save the trained tokenizer
output_path = join(args.out, f"{args.name}.json")
tokenizer.save(output_path)
print(f"Trained tokenizer saved to: {output_path}")

# Create an inference version without pre-tokenizer
# First save to a temporary file
temp_tokenizer_path = join(args.out, "temp_tokenizer.json")
tokenizer.save(temp_tokenizer_path)

# Read the JSON
with open(temp_tokenizer_path, "r") as f:
    tokenizer_data = json.load(f)

# Remove pre-tokenizer field if present
if "pre_tokenizer" in tokenizer_data:
    del tokenizer_data["pre_tokenizer"]

# Write modified tokenizer to inference file
inference_path = join(args.out, f"{args.name}_inference.json")
with open(inference_path, "w") as f:
    json.dump(tokenizer_data, f, indent=2)

# Clean up temp file
os.remove(temp_tokenizer_path)

print(f"Inference-ready tokenizer (no pre-tokenizer) saved to: {inference_path}")

# Test encoding with inference tokenizer
tokenizer = Tokenizer.from_file(inference_path)
example = "Training BPE with multi-word tokens is very easy"
print(f"\nTest encoding: {tokenizer.encode(example).tokens}")