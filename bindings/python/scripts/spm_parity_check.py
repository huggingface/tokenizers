import tokenizers
from argparse import ArgumentParser
import sentencepiece as spm
import json
import os


def main():
    parser = ArgumentParser("SentencePiece parity checker")
    parser.add_argument(
        "--input-file", "-i", type=str, required=True, help="Which files do you want to train from",
    )
    parser.add_argument(
        "--model-prefix", type=str, default="spm_parity", help="Model prefix for spm_train",
    )
    parser.add_argument(
        "--vocab-size", "-v", type=int, default=8000, help="Vocab size for spm_train",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Instead of checking the encoder part, we check the trainer part",
    )

    args = parser.parse_args()

    spm.SentencePieceTrainer.Train(
        f"--input={args.input_file} --model_prefix={args.model_prefix}"
        f" --character_coverage=1.0"
        f" --max_sentence_length=40000"
        f" --num_threads=1"
        f" --vocab_size={args.vocab_size}"
    )

    try:
        if args.train:
            check_train(args)
        else:
            check_encode(args)
    finally:
        os.remove(f"{args.model_prefix}.model")
        os.remove(f"{args.model_prefix}.vocab")


def check_train(args):
    sp = spm.SentencePieceProcessor()
    model_filename = f"{args.model_prefix}.model"
    sp.Load(model_filename)

    tokenizer = tokenizers.SentencePieceUnigramTokenizer()
    tokenizer.train(args.input_file, show_progress=False)

    spm_tokens = 0
    tokenizer_tokens = 0

    with open(args.input_file, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            ids = sp.EncodeAsIds(line)

            encoded = tokenizer.encode(line)

            spm_tokens += len(ids)
            tokenizer_tokens += len(encoded.ids)

    vocab = [0 for i in range(args.vocab_size)]
    spm_vocab = [0 for i in range(args.vocab_size)]

    for token, index in tokenizer.get_vocab().items():
        vocab[index] = token

    for i in range(args.vocab_size):
        spm_vocab[i] = sp.id_to_piece(i)

    # 0 is unk in tokenizers, 0, 1, 2 are unk bos, eos in spm by default.
    for i, (token, spm_token) in enumerate(zip(vocab[1:], spm_vocab[3:])):
        if token != spm_token:
            print(f"First different token is token {i} ({token} != {spm_token})")
            break

    print(f"Tokenizer used {tokenizer_tokens}, where spm used {spm_tokens}")
    assert (
        tokenizer_tokens < spm_tokens
    ), "Our trainer should be at least more efficient than the SPM one"
    print("Ok our trainer is at least more efficient than the SPM one")


def check_encode(args):
    sp = spm.SentencePieceProcessor()
    model_filename = f"{args.model_prefix}.model"
    sp.Load(model_filename)

    vocab_filename = f"{args.model_prefix}.json"

    vocab = [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.piece_size())]

    data = {"unk_id": sp.unk_id(), "vocab": vocab}

    with open(vocab_filename, "w") as f:
        json.dump(data, f, indent=4)

    try:
        tok = tokenizers.SentencePieceUnigramTokenizer(vocab_filename)
        with open(args.input_file, "r") as f:
            for i, line in enumerate(f):
                line = line.strip()
                ids = sp.EncodeAsIds(line)

                encoded = tok.encode(line)

                if ids != encoded.ids:
                    # Encoding can be the same with same result AAA -> A + AA vs AA + A
                    # We can check that we use at least exactly the same number of tokens.
                    assert len(ids) == len(
                        encoded.ids
                    ), f"{len(ids)} != {len(encoded.ids)} \nTokenizer: {encoded.ids}\nSpm:       {ids}"
                    continue

                assert ids == encoded.ids, f"line {i}: {line} : {ids} != {encoded.ids}"
    finally:
        os.remove(vocab_filename)


if __name__ == "__main__":
    main()
