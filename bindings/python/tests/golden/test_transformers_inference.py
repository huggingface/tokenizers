"""Layer 3: transformers inference.

The tokenizer feeds prompts into a model and turns its output ids back into
text. Models are tiny and random (see conftest), so the assertions check what
the tokenizer is responsible for — prompt round-trips, shapes, masks — never
generation quality.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from .conftest import TINY_BERT_MLM, TINY_GPT2


def test_batched_generate_round_trips_the_prompt():
    # Decoder-only generation pads on the left so prompts sit flush against
    # the generated positions. GPT-2 has no pad token; reusing EOS is the
    # standard recipe.
    tok = AutoTokenizer.from_pretrained(TINY_GPT2, padding_side="left")
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(TINY_GPT2)
    prompts = ["Hello world", "The quick brown fox jumps"]

    inputs = tok(prompts, padding=True, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    texts = tok.batch_decode(out, skip_special_tokens=True)

    assert out.shape == (2, inputs["input_ids"].shape[1] + 5)
    for prompt, text in zip(prompts, texts):
        assert text.startswith(prompt)


def test_text_generation_pipeline():
    # pipeline() bundles tokenizer + model + decoding into one call.
    generate = pipeline("text-generation", model=TINY_GPT2)

    result = generate("Hello world", max_new_tokens=5, do_sample=False)

    assert result[0]["generated_text"].startswith("Hello world")


def test_fill_mask_pipeline():
    # Fill-mask leans on tokenizer internals: locating the mask token's
    # position and decoding single-token candidates back to strings.
    fill = pipeline("fill-mask", model=TINY_BERT_MLM)

    candidates = fill("Paris is the [MASK] of France.")

    assert len(candidates) == 5
    for candidate in candidates:
        assert candidate.keys() == {"score", "token", "token_str", "sequence"}
        assert "[MASK]" not in candidate["sequence"]
    assert candidates[0]["sequence"].startswith("paris is the")
