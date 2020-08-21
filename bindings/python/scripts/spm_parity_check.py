import tokenizers
from argparse import ArgumentParser
import sentencepiece as spm
import json


def main():
    parser = ArgumentParser("SentencePiece parity checker")
    parser.add_argument(
        "--input-file",
        "-i",
        type=str,
        required=True,
        help="Which files do you want to train from",
    )
    parser.add_argument(
        "--model-prefix",
        type=str,
        default="spm_parity",
        help="Model prefix for spm_train",
    )
    parser.add_argument(
        "--vocab-size", "-v", type=int, default=8000, help="Vocab size for spm_train",
    )

    args = parser.parse_args()

    spm.SentencePieceTrainer.Train(
        f"--input={args.input_file} --model_prefix={args.model_prefix}"
        f" --vocab_size={args.vocab_size}"
    )

    sp = spm.SentencePieceProcessor()
    model_filename = f"{args.model_prefix}.model"
    sp.Load(model_filename)

    vocab_filename = f"{args.model_prefix}.json"

    vocab = [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.piece_size())]

    data = {"unk_id": sp.unk_id(), "vocab": vocab}

    with open(vocab_filename, "w") as f:
        json.dump(data, f, indent=4)

    tok = tokenizers.SentencePieceUnigramTokenizer(vocab_filename)
    with open(args.input_file, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            ids = sp.EncodeAsIds(line)

            encoded = tok.encode(line)

            if ids != encoded.ids:
                # Encoding can be the same with same result AAA -> A + AA vs AA + A
                # We just check this does not cover unk tokens
                if len(ids) != len(encoded.ids):
                    N = len(ids)
                    M = len(encoded.ids)
                    first_index_error = [
                        i for i in range(min(N, M)) if ids[i] != encoded.ids[i]
                    ][0]
                    last_index_error = [
                        min(N, M) - i
                        for i in range(min(N, M))
                        if ids[-i - 1] != encoded.ids[-i - 1]
                    ][0]
                    print(ids[first_index_error : last_index_error + 1])
                    print(encoded.ids[first_index_error : last_index_error + 1])
                    import ipdb

                    ipdb.set_trace()
                assert len(ids) == len(encoded.ids)
                continue

            assert ids == encoded.ids, f"line {i}: {line} : {ids} != {encoded.ids}"


if __name__ == "__main__":
    main()
