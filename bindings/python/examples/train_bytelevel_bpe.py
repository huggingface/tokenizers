import argparse
import glob

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers


parser = argparse.ArgumentParser()
parser.add_argument("--files",
                    default=None,
                    metavar="path",
                    type=str,
                    required=True,
                    help="The files to use as training; accept '**/*.txt' type of patterns \
                          if enclosed in quotes")
parser.add_argument("--out",
                    default="./",
                    type=str,
                    help="Path to the output directory, where the files will be saved")
parser.add_argument("--name",
                    default="bpe-bytelevel",
                    type=str,
                    help="The name of the output vocab files")
args = parser.parse_args()

files = glob.glob(args.files)
if not files:
    print(f"File does not exist: {args.files}")
    exit(1)


# Initialize an empty tokenizer
tokenizer = Tokenizer(models.BPE.empty())

# Customize pre-tokenization and decoding
tokenizer.with_pre_tokenizer(pre_tokenizers.ByteLevel.new(add_prefix_space=False))
tokenizer.with_decoder(decoders.ByteLevel.new())

# And then train
trainer = trainers.BpeTrainer.new(
    vocab_size=50000,
    min_frequency=2,
    show_progress=True,
    special_tokens=[ "<s>", "<pad>", "</s" ],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
)
tokenizer.train(trainer, files)

# Save the files
tokenizer.model.save(args.out, args.name)
