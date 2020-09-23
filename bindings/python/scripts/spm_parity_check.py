import tokenizers
from argparse import ArgumentParser
import sentencepiece as spm
from collections import Counter
import json
import os
import datetime

try:
    from termcolor import colored

    has_color = True
except Exception:
    has_color = False


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
        "--model-file",
        "-m",
        type=str,
        required=False,
        default=None,
        help="Use a pretrained token file",
    )
    parser.add_argument(
        "--model-prefix",
        type=str,
        default="spm_parity",
        help="Model prefix for spm_train",
    )
    parser.add_argument(
        "--vocab-size",
        "-v",
        type=int,
        default=8000,
        help="Vocab size for spm_train",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbosity",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Instead of checking the encoder part, we check the trainer part",
    )
    parser.add_argument(
        "--from-spm",
        action="store_true",
        help="Directly load the spm file with it's own normalizer",
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


def check_diff(spm_diff, tok_diff, sp, tok):
    if spm_diff == list(reversed(tok_diff)):
        # AAA -> AA+A vs A+AA case.
        return True
    # elif len(spm_diff) == len(tok_diff) and tok.decode(spm_diff) == tok.decode(
    #     tok_diff
    # ):
    #     # Second order OK
    #     # Barrich -> Barr + ich vs Bar + rich
    #     return True
    spm_reencoded = sp.encode(sp.decode(spm_diff))
    tok_reencoded = tok.encode(tok.decode(spm_diff)).ids
    if spm_reencoded != spm_diff and spm_reencoded == tok_reencoded:
        # Type 3 error.
        # Snehagatha ->
        #       Sne, h, aga, th, a
        #       Sne, ha, gat, ha
        # Encoding the wrong with sp does not even recover what spm gave us
        # It fits tokenizer however...
        return True
    return False


def check_details(line, spm_ids, tok_ids, sp, tok):
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

    if check_diff(spm_diff, tok_diff, sp, tok):
        return True

    if last - first > 5:
        # We might have twice a single problem, attempt to subdivide the disjointed tokens into smaller problems
        spms = Counter(spm_ids[first:last])
        toks = Counter(tok_ids[first:last])

        removable_tokens = {
            spm_ for (spm_, si) in spms.items() if toks.get(spm_, 0) == si
        }
        min_width = 3
        for i in range(last - first - min_width):
            if all(
                spm_ids[first + i + j] in removable_tokens for j in range(min_width)
            ):
                possible_matches = [
                    k
                    for k in range(last - first - min_width)
                    if tok_ids[first + k : first + k + min_width]
                    == spm_ids[first + i : first + i + min_width]
                ]
                for j in possible_matches:
                    if check_diff(
                        spm_ids[first : first + i], tok_ids[first : first + j], sp, tok
                    ) and check_details(
                        line,
                        spm_ids[first + i : last],
                        tok_ids[first + j : last],
                        sp,
                        tok,
                    ):
                        return True

    print(f"Spm: {[tok.decode([spm_ids[i]]) for i in range(first, last)]}")
    try:
        print(f"Tok: {[tok.decode([tok_ids[i]]) for i in range(first, last)]}")
    except Exception:
        pass

    ok_start = tok.decode(spm_ids[:first])
    ok_end = tok.decode(spm_ids[last:])
    wrong = tok.decode(spm_ids[first:last])
    print()
    if has_color:
        print(
            f"{colored(ok_start, 'grey')}{colored(wrong, 'red')}{colored(ok_end, 'grey')}"
        )
    else:
        print(wrong)
    return False


def check_encode(args):
    sp = spm.SentencePieceProcessor()
    sp.Load(args.model_file)

    if args.from_spm:
        tok = tokenizers.SentencePieceUnigramTokenizer.from_spm(args.model_file)
    else:
        vocab = [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.piece_size())]
        unk_id = sp.unk_id()
        tok = tokenizers.SentencePieceUnigramTokenizer(vocab, unk_id)

    perfect = 0
    imperfect = 0
    wrong = 0
    now = datetime.datetime.now
    spm_total_time = datetime.timedelta(seconds=0)
    tok_total_time = datetime.timedelta(seconds=0)
    with open(args.input_file, "r", encoding="utf-8-sig") as f:
        for i, line in enumerate(f):
            line = line.strip()

            start = now()
            ids = sp.EncodeAsIds(line)
            spm_time = now()

            encoded = tok.encode(line)
            tok_time = now()

            spm_total_time += spm_time - start
            tok_total_time += tok_time - spm_time

            if args.verbose:
                if i % 10000 == 0:
                    print(
                        f"({perfect} / {imperfect} / {wrong} ----- {perfect + imperfect + wrong})"
                    )
                    print(f"SPM: {spm_total_time} - TOK: {tok_total_time}")

            if ids != encoded.ids:
                if check_details(line, ids, encoded.ids, sp, tok):
                    imperfect += 1
                    continue
                else:
                    wrong += 1
            else:
                perfect += 1

            assert ids == encoded.ids, f"line {i}: {line} : {ids} != {encoded.ids}"

    print(f"({perfect} / {imperfect} / {wrong} ----- {perfect + imperfect + wrong})")
    total = perfect + imperfect + wrong
    print(
        f"Accuracy {perfect * 100 / total:.2f} Slowdown : {tok_total_time/ spm_total_time:.2f}"
    )


if __name__ == "__main__":
    main()
