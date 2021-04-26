import argparse
import pprint
from typing import Any
import torch
from transformers import AutoTokenizer, BartForConditionalGeneration
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def pretty_convert(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return x.tolist()
    else:
        return x


def pretty_print(x: dict) -> None:
    y = {k: pretty_convert(v) for k, v in x.items()}
    pp = pprint.PrettyPrinter(indent=4, compact=True, width=120)
    pp.pprint(y)


def main(args=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/bart-base")
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    pretty_print(args.__dict__)

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(args.model)

    tokenized_inputs = tokenizer(
        [args.input],
        padding="longest",
        return_tensors="pt",
        add_special_tokens=True,
        return_attention_mask=False,
    )
    pretty_print(tokenized_inputs)
    
    back_decoded = tokenizer.batch_decode(
        tokenized_inputs["input_ids"],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    pretty_print({"back_decoded": back_decoded})

if __name__ == "__main__":
    main()