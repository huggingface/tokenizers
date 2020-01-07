import argparse
import glob

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, normalizers


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
                    default="bert-wordpiece",
                    type=str,
                    help="The name of the output vocab files")
args = parser.parse_args()

files = glob.glob(args.files)
if not files:
    print(f"File does not exist: {args.files}")
    exit(1)


# Initialize an empty tokenizer
tokenizer = Tokenizer(models.WordPiece.empty())

# Customize all the steps
tokenizer.normalizer = normalizers.BertNormalizer.new(
    clean_text=True,
    handle_chinese_chars=True,
    strip_accents=True,
    lowercase=True,
)
tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer.new()
tokenizer.decoder = decoders.WordPiece.new()

# And then train
trainer = trainers.WordPieceTrainer.new(
    vocab_size=50000,
    min_frequency=2,
    show_progress=True,
    special_tokens=[ "<s>", "<unk>", "<pad>", "</s>" ],
    limit_alphabet=1000,
    continuing_subword_prefix="##"
)
tokenizer.train(trainer, files)

# Save the files
tokenizer.model.save(args.out, args.name)

