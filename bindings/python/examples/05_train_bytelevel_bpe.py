"""Train a byte-level BPE tokenizer (GPT-2 style) from text files: ByteLevel
pre-tokenization + BpeTrainer seeded with the byte alphabet (the recipe
`ByteLevelBPETokenizer` bundled in tokenizers 0.x). The encode pipeline does
not support `add_prefix_space`, so it is always off."""

import argparse
import glob
import tempfile
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, trainers

DEFAULT_CORPUS = Path(__file__).resolve().parents[3] / "tokenizers" / "data" / "big.txt"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--files",
    default=str(DEFAULT_CORPUS),
    metavar="path",
    help="training files; '**/*.txt' patterns work if enclosed in quotes",
)
parser.add_argument("--out", default=tempfile.mkdtemp(), help="output directory")
parser.add_argument("--name", default="bpe-bytelevel", help="name of the saved tokenizer file")
args = parser.parse_args()

files = glob.glob(args.files)
assert files, f"no files match {args.files}"

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

tokenizer.train(
    files,
    trainer=trainers.BpeTrainer(
        vocab_size=10000,
        min_frequency=2,
        special_tokens=["<s>", "<pad>", "</s>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    ),
)

out = Path(args.out) / f"{args.name}.json"
tokenizer.save(out)
print(f"saved {out} ({tokenizer.get_vocab_size()} tokens)")

reloaded = Tokenizer.from_file(out)
ids = reloaded.encode_ids("Training ByteLevel BPE is very easy", add_special_tokens=False)
print([reloaded.id_to_token(i) for i in ids])
