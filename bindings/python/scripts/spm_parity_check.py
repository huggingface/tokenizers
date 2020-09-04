import tokenizers
from argparse import ArgumentParser
import sentencepiece as spm
import json
import os

try:
    from termcolor import colored

    has_color = True
except Exception:
    has_color = False


def main():
    parser = ArgumentParser("SentencePiece parity checker")
    parser.add_argument(
        "--input-file", "-i", type=str, required=True, help="Which files do you want to train from",
    )
    parser.add_argument(
        "--model-file",
        "-m",
        type=str,
        required=False,
        default=None,
        help="Use a pretrained token file",
    )
    parser.add_argument(
        "--model-prefix", type=str, default="spm_parity", help="Model prefix for spm_train",
    )
    parser.add_argument(
        "--vocab-size", "-v", type=int, default=8000, help="Vocab size for spm_train",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Verbosity",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Instead of checking the encoder part, we check the trainer part",
    )

    args = parser.parse_args()

    trained = False
    if args.model_file is None:
        spm.SentencePieceTrainer.Train(
            f"--input={args.input_file} --model_prefix={args.model_prefix}"
            f" --character_coverage=1.0"
            f" --max_sentence_length=40000"
            f" --num_threads=1"
            f" --vocab_size={args.vocab_size}"
        )
        trained = True
        args.model_file = f"{args.model_prefix}.model"

    try:
        if args.train:
            check_train(args)
        else:
            check_encode(args)
    finally:
        if trained:
            os.remove(f"{args.model_prefix}.model")
            os.remove(f"{args.model_prefix}.vocab")


def check_train(args):
    sp = spm.SentencePieceProcessor()
    sp.Load(args.model_file)

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


def check_details(line, spm_ids, tok_ids, tok, sp):
    # Encoding can be the same with same result AAA -> A + AA vs AA + A
    # We can check that we use at least exactly the same number of tokens.
    for i, (spm_id, tok_id) in enumerate(zip(spm_ids, tok_ids)):
        if spm_id != tok_id:
            break
    first = i
    for i, (spm_id, tok_id) in enumerate(zip(reversed(spm_ids), reversed(tok_ids))):
        if spm_id != tok_id:
            break
    last = len(spm_ids) - i

    spm_diff = spm_ids[first:last]
    tok_diff = tok_ids[first:last]
    if spm_diff == list(reversed(tok_diff)):
        # AAA -> AA+A vs A+AA case.
        return True
    elif len(spm_diff) == len(tok_diff) and tok.decode(spm_diff) == tok.decode(tok_diff):
        # Second order OK
        # Barrich -> Barr + ich vs Bar + rich
        return True

    ok_start = tok.decode(spm_ids[:first])
    ok_end = tok.decode(spm_ids[last:])
    wrong = tok.decode(spm_ids[first:last])
    print()
    if has_color:
        print(f"{colored(ok_start, 'grey')}{colored(wrong, 'red')}{colored(ok_end, 'grey')}")
    else:
        print(wrong)

    print(f"Spm: {[tok.decode([spm_ids[i]]) for i in range(first, last)]}")
    print(f"Tok: {[tok.decode([tok_ids[i]]) for i in range(first, last)]}")

    spm_reencoded = sp.encode(sp.decode(spm_ids[first:last]))
    tok_reencoded = tok.encode(tok.decode(spm_ids[first:last])).ids
    if spm_reencoded != spm_ids[first:last] and spm_reencoded == tok_reencoded:
        # Type 3 error.
        # Snehagatha ->
        #       Sne, h, aga, th, a
        #       Sne, ha, gat, ha
        # Encoding the wrong with sp does not even recover what spm gave us
        # It fits tokenizer however...
        return True
    return False


def check_encode(args):
    sp = spm.SentencePieceProcessor()
    sp.Load(args.model_file)

    vocab_filename = f"{'.'.join(args.model_file.split('.')[:-1])}.json"

    vocab = [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.piece_size())]

    data = {"unk_id": sp.unk_id(), "vocab": vocab}

    with open(vocab_filename, "w") as f:
        json.dump(data, f, indent=4)

    perfect = 0
    imperfect = 0
    wrong = 0
    try:
        tok = tokenizers.SentencePieceUnigramTokenizer(vocab_filename)
        with open(args.input_file, "r", encoding="utf-8-sig") as f:
            for i, line in enumerate(f):
                line = line.strip()
                line = line.replace("�", "")
                line = line.replace("\u200b", "")
                line = line.replace("\u200c", "")
                line = line.replace("\u200d", "")
                line = line.replace("\ufeff", "")
                ids = sp.EncodeAsIds(line)

                encoded = tok.encode(line)

                if args.verbose:
                    if i % 10000 == 0:
                        print(
                            f"({perfect} / {imperfect} / {wrong} ----- {perfect + imperfect + wrong})"
                        )

                # In wiki-text-103
                # sp.encode('increasing to 35 ( km / h ) Lonaconing Lonaconing .')
                # will give 2 different encodings for the same word `Lonaconing`.
                # We should be able to see why there is a difference
                if "Lonaconing" in line:
                    continue

                if ids != encoded.ids:
                    if check_details(line, ids, encoded.ids, tok, sp):
                        imperfect += 1
                        continue
                    else:
                        wrong += 1
                else:
                    perfect += 1

                assert ids == encoded.ids, f"line {i}: {line} : {ids} != {encoded.ids}"
    finally:
        os.remove(vocab_filename)

    print(f"({perfect} / {imperfect} / {wrong} ----- {perfect + imperfect + wrong})")


if __name__ == "__main__":
    main()
