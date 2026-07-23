"""Layer 5: production flows.

Where the recipe tests pinpoint single features, each test here plays out one
whole production story from the three domains where transformers earns its
keep — LLM chat serving, retrieval (RAG), and supervised fine-tuning — the
way serving stacks (vLLM, TGI, `transformers serve`), embedding pipelines
(sentence-transformers, text splitters), and SFT trainers (TRL) drive the
tokenizer. Models stay tiny and random (see conftest): the assertions pin the
tokenizer's side of the story, not model quality.
"""

import math
from threading import Thread

import datasets
import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    TextIteratorStreamer,
    Trainer,
    TrainingArguments,
)

from .conftest import TINY_BERT_MLM, TINY_GPT2

# The ChatML conversation format used by the SmolLM/Qwen instruct families.
CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def chat_tokenizer(**kwargs):
    # A serving or training stack customizes its tokenizer once at startup:
    # chat template, the template's control tokens as specials, a pad token.
    tok = AutoTokenizer.from_pretrained(TINY_GPT2, **kwargs)
    tok.chat_template = CHATML_TEMPLATE
    tok.add_special_tokens({"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]})
    tok.pad_token = tok.eos_token
    return tok


def test_chat_serving_flow():
    tok = chat_tokenizer(padding_side="left")  # prompts flush against generation
    model = AutoModelForCausalLM.from_pretrained(TINY_GPT2)
    model.resize_token_embeddings(len(tok))

    # The control tokens registered as specials encode to single ids and are
    # stripped from decoded replies.
    im_start = tok.convert_tokens_to_ids("<|im_start|>")
    assert tok("<|im_start|>", add_special_tokens=False)["input_ids"] == [im_start]

    # Fit the conversation into the context budget by dropping the oldest
    # exchanges; the system prompt always survives.
    def n_tokens(conversation):
        text = tok.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        return len(tok(text, add_special_tokens=False)["input_ids"])

    conversation = [{"role": "system", "content": "You are a concise assistant."}]
    for i in range(3):
        conversation += [
            {"role": "user", "content": f"Tell me an interesting fact, number {i}, please."},
            {"role": "assistant", "content": f"Here is interesting fact number {i} for you."},
        ]
    conversation.append({"role": "user", "content": "Now summarize them all."})

    budget = n_tokens(conversation) - 1
    while n_tokens(conversation) > budget:
        del conversation[1:3]
    assert n_tokens(conversation) <= budget
    assert conversation[0]["role"] == "system"
    assert len(conversation) == 6  # exactly one exchange dropped

    # Serve a batch of two conversations, left-padded to one rectangle.
    other = [{"role": "user", "content": "Hi!"}]
    prompts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in (conversation, other)]
    inputs = tok(prompts, padding=True, return_tensors="pt")
    prompt_len = inputs["input_ids"].shape[1]
    out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    replies = tok.batch_decode(out[:, prompt_len:], skip_special_tokens=True)
    assert len(replies) == 2
    assert all(isinstance(reply, str) for reply in replies)

    # Stream the same request token by token: the concatenated stream must
    # equal the one-shot decode, however tokens split mid-word or mid-byte.
    single = tok(prompts[0], return_tensors="pt")
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    worker = Thread(
        target=model.generate,
        kwargs={**single, "max_new_tokens": 8, "do_sample": False, "streamer": streamer},
    )
    worker.start()
    streamed = "".join(streamer)
    worker.join()
    one_shot = model.generate(**single, max_new_tokens=8, do_sample=False)
    assert streamed == tok.decode(one_shot[0, single["input_ids"].shape[1] :], skip_special_tokens=True)


def test_rag_chunk_embed_retrieve_flow():
    tok = AutoTokenizer.from_pretrained(TINY_BERT_MLM)
    encoder = AutoModel.from_pretrained(TINY_BERT_MLM)
    document = " ".join(f"Sentence number {i} talks at length about topic {i % 7}." for i in range(80))

    # Chunk by token budget, keeping each chunk's char span so retrieval can
    # point back into the original document.
    encoding = tok(document, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoding["offset_mapping"]
    budget = 48
    spans = [
        (window[0][0], window[-1][1]) for window in (offsets[i : i + budget] for i in range(0, len(offsets), budget))
    ]
    chunks = [document[start:end] for start, end in spans]

    # The spans tile the document: in order, non-overlapping, nothing lost
    # but the whitespace between chunks.
    assert len(chunks) > 3
    assert spans[0][0] == 0
    assert spans[-1][1] == len(document)
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))

    # Embed chunks and query the sentence-transformers way: mean-pool hidden
    # states over the attention mask, so padding never dilutes the vector.
    def embed(texts):
        batch = tok(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
        with torch.no_grad():
            hidden = encoder(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1)
        vectors = (hidden * mask).sum(1) / mask.sum(1)
        return torch.nn.functional.normalize(vectors, dim=1)

    chunk_vectors = embed(chunks)
    query_vector = embed(["Which sentence talks about topic 3?"])
    assert torch.isfinite(chunk_vectors).all()

    # Whatever the (random) model ranks first, the service returns an exact
    # substring of the source document.
    best = int((chunk_vectors @ query_vector.T).argmax())
    start, end = spans[best]
    assert document[start:end] == chunks[best]


def test_sft_finetuning_flow(tmp_path):
    tok = chat_tokenizer()
    model = AutoModelForCausalLM.from_pretrained(TINY_GPT2)
    model.resize_token_embeddings(len(tok))

    raw = datasets.Dataset.from_list(
        [{"prompt": f"Question number {i}?", "response": f"Answer number {i}."} for i in range(8)]
    )

    def to_features(example):
        # Prompt and reply are tokenized separately then concatenated — the
        # TRL recipe — so no BPE merge can blur the boundary; the loss covers
        # the reply only.
        messages = [{"role": "user", "content": example["prompt"]}]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        reply_ids = tok(example["response"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
        return {
            "input_ids": prompt_ids + reply_ids,
            "attention_mask": [1] * (len(prompt_ids) + len(reply_ids)),
            "labels": [-100] * len(prompt_ids) + reply_ids,
        }

    # num_proc=2 pickles the tokenizer into worker processes, like any real
    # preprocessing job over a large dataset.
    features = raw.map(to_features, num_proc=2, remove_columns=raw.column_names)

    first = features[0]
    boundary = first["labels"].count(-100)
    assert 0 < boundary < len(first["labels"])
    assert first["labels"][boundary:] == first["input_ids"][boundary:]

    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir=str(tmp_path), max_steps=3, per_device_train_batch_size=4, report_to=[]),
        train_dataset=features,
        # Pads ragged input_ids with the pad token and labels with -100.
        data_collator=DataCollatorForSeq2Seq(tok, label_pad_token_id=-100),
    )
    result = trainer.train()

    assert math.isfinite(result.training_loss)
