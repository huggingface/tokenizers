"""Layer 4: transformers training.

The tokenizer's job in a training loop: turn a labeled corpus into ragged id
lists up front, then let a data collator pad each batch on the fly. A few
optimizer steps on a tiny random model prove the loop consumes them.
"""

import math

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from .conftest import TINY_BERT_CLS, TINY_GPT2


def test_data_collator_pads_a_ragged_batch():
    # DataCollatorWithPadding is tokenizer.pad() in disguise: ragged encodings
    # in, rectangular tensors out.
    tok = AutoTokenizer.from_pretrained(TINY_BERT_CLS)
    features = [tok(text) for text in ("One two", "one two three four five")]

    batch = DataCollatorWithPadding(tok)(features)

    assert batch["input_ids"].shape == batch["attention_mask"].shape
    assert batch["input_ids"][0, -1] == tok.pad_token_id
    assert batch["attention_mask"][0, -1] == 0
    assert batch["attention_mask"][1].tolist() == [1] * batch["input_ids"].shape[1]


def test_causal_lm_collator_masks_padding_in_labels():
    # For causal LM the labels are the input ids, except padding must not
    # contribute to the loss: it becomes -100.
    tok = AutoTokenizer.from_pretrained(TINY_GPT2)
    tok.pad_token = tok.eos_token
    features = [tok(text) for text in ("Tiny", "a longer line of text")]

    batch = DataCollatorForLanguageModeling(tok, mlm=False)(features)

    padded = batch["attention_mask"] == 0
    assert (batch["labels"][padded] == -100).all()
    assert (batch["labels"][~padded] == batch["input_ids"][~padded]).all()


def test_trainer_runs_a_few_steps(tmp_path):
    texts = ["a delightful film", "an utter disappointment", "warm and funny", "dull beyond belief"] * 4
    labels = [1, 0, 1, 0] * 4

    tok = AutoTokenizer.from_pretrained(TINY_BERT_CLS)
    model = AutoModelForSequenceClassification.from_pretrained(TINY_BERT_CLS)
    encodings = tok(texts, truncation=True, max_length=32)

    class SentimentDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(labels)

        def __getitem__(self, i):
            return {
                "input_ids": encodings["input_ids"][i],
                "attention_mask": encodings["attention_mask"][i],
                "labels": labels[i],
            }

    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir=str(tmp_path), max_steps=3, per_device_train_batch_size=4, report_to=[]),
        train_dataset=SentimentDataset(),
        data_collator=DataCollatorWithPadding(tok),
    )
    result = trainer.train()

    assert math.isfinite(result.training_loss)
