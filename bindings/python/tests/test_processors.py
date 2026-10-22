import copy
import json
import pickle

import pytest

from tokenizers import processors

TEXT = "Hello world, how are you?"
PAIR = ("Hello world", "how are you?")
BERT_SPECIALS = [("[CLS]", 1), ("[SEP]", 2)]
SINGLE_A = '"single":[{"Sequence":{"id":"A","type_id":0}}]'
PAIR_A_B1 = '"pair":[{"Sequence":{"id":"A","type_id":0}},{"Sequence":{"id":"B","type_id":1}}]'
CLS = '{"SpecialToken":{"id":"[CLS]","type_id":0}}'
SEP = '{"SpecialToken":{"id":"[SEP]","type_id":0}}'
A = '{"Sequence":{"id":"A","type_id":0}}'
BERT_SPECIALS_JSON = (
    '"special_tokens":{"[CLS]":{"id":"[CLS]","ids":[1],"tokens":["[CLS]"]},'
    '"[SEP]":{"id":"[SEP]","ids":[2],"tokens":["[SEP]"]}}'
)


# The JSON 0.23 writes for each, key order included.
@pytest.mark.parametrize(
    ("fixture", "post_processor", "written"),
    [
        (
            "bert_spec",
            processors.TemplateProcessing(),
            f'{{"type":"TemplateProcessing",{SINGLE_A},{PAIR_A_B1},"special_tokens":{{}}}}',
        ),
        (
            "bert_spec",
            processors.TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=BERT_SPECIALS),
            f'{{"type":"TemplateProcessing","single":[{CLS},{A},{SEP}],{PAIR_A_B1},{BERT_SPECIALS_JSON}}}',
        ),
        (
            "bert_spec",
            processors.TemplateProcessing(
                single="[CLS] $A [SEP]", pair="[CLS] $A:0 [SEP] $B:1 [SEP]:1", special_tokens=BERT_SPECIALS
            ),
            f'{{"type":"TemplateProcessing","single":[{CLS},{A},{SEP}],"pair":[{CLS},{A},{SEP},'
            '{"Sequence":{"id":"B","type_id":1}},{"SpecialToken":{"id":"[SEP]","type_id":1}}],'
            f"{BERT_SPECIALS_JSON}}}",
        ),
        (
            "bert_spec",
            processors.TemplateProcessing(
                single=["[CLS]", "$A"], pair=["[CLS]", "$A", "$B:1"], special_tokens=BERT_SPECIALS
            ),
            f'{{"type":"TemplateProcessing","single":[{CLS},{A}],"pair":[{CLS},{A},'
            f'{{"Sequence":{{"id":"B","type_id":1}}}}],{BERT_SPECIALS_JSON}}}',
        ),
        # sentence-transformers writes `$0` for the first sequence.
        (
            "bert_spec",
            processors.TemplateProcessing(single="$0", pair="$0 $B:1", special_tokens=[]),
            f'{{"type":"TemplateProcessing",{SINGLE_A},{PAIR_A_B1},"special_tokens":{{}}}}',
        ),
        (
            "bert_spec",
            processors.TemplateProcessing(special_tokens=[("[SEP]", 2), ("[CLS]", 9), ("[CLS]", 1)]),
            f'{{"type":"TemplateProcessing",{SINGLE_A},{PAIR_A_B1},{BERT_SPECIALS_JSON}}}',
        ),
        (
            "bert_spec",
            processors.BertProcessing(("[SEP]", 2), ("[CLS]", 1)),
            '{"type":"BertProcessing","sep":["[SEP]",2],"cls":["[CLS]",1]}',
        ),
        (
            "gpt2_spec",
            processors.RobertaProcessing(("</s>", 2), ("<s>", 0)),
            '{"type":"RobertaProcessing","sep":["</s>",2],"cls":["<s>",0],"trim_offsets":true,"add_prefix_space":true}',
        ),
        (
            "gpt2_spec",
            processors.RobertaProcessing(("</s>", 2), ("<s>", 0), False, False),
            '{"type":"RobertaProcessing","sep":["</s>",2],"cls":["<s>",0],"trim_offsets":false,'
            '"add_prefix_space":false}',
        ),
        (
            "gpt2_spec",
            processors.ByteLevel(),
            '{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true}',
        ),
        (
            "gpt2_spec",
            processors.ByteLevel(trim_offsets=False),
            '{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":false,"use_regex":true}',
        ),
    ],
    ids=lambda param: type(param).__name__ if isinstance(param, processors.PostProcessor) else None,
)
def test_post_processor_json(request, fixture, post_processor, written):
    assert post_processor.__getstate__().decode() == written
    assert post_processor
    twin = pickle.loads(pickle.dumps(post_processor))
    assert type(twin) is type(post_processor)
    tokenizer = copy.deepcopy(request.getfixturevalue(fixture))
    tokenizer.post_processor = post_processor
    assert type(tokenizer.post_processor) is type(post_processor)
    compiled = tokenizer.compile()
    compiled.decode(compiled.encode(PAIR).ids)
    # The core compiles Bert, Roberta and ByteLevel into a template, but the file keeps them as given.
    assert json.loads(tokenizer.to_str())["post_processor"] == json.loads(written)


def special_tokens(tokenizer):
    return [(token["content"], token["id"]) for token in json.loads(tokenizer.to_str())["added_tokens"]]


# transformers rebuilds the template from the tokenizer's own bos/eos tokens and ids.
def test_template_processing_encodes_like_the_file_it_was_rebuilt_from(bert, bert_spec):
    tokenizer = copy.deepcopy(bert_spec)
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1", special_tokens=special_tokens(bert_spec)
    )
    rebuilt, declared = tokenizer.compile().encode(PAIR), bert.encode(PAIR)
    assert (rebuilt.ids, rebuilt.type_ids) == (declared.ids, declared.type_ids)


def test_bert_processing_encodes_like_its_template(bert, bert_spec):
    tokenizer = copy.deepcopy(bert_spec)
    tokenizer.post_processor = processors.BertProcessing(("[SEP]", 2), ("[CLS]", 1))
    rebuilt, declared = tokenizer.compile().encode(PAIR), bert.encode(PAIR)
    assert (rebuilt.ids, rebuilt.type_ids) == (declared.ids, declared.type_ids)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"single": "$C"}, 'Cannot build Piece from string "$C"'),
        ({"single": "$A:x"}, 'Cannot build Piece from string "$A:x"'),
        (
            {"single": "[CLS] $A [X] [Y]", "special_tokens": BERT_SPECIALS},
            "Missing SpecialToken(s) with id(s) `[X], [Y]`",
        ),
        # 0.23 reads the empty piece before a leading space as a special token named "".
        ({"single": " $A"}, "Missing SpecialToken(s) with id(s) ``"),
        ({"pair": "$A [SEP]", "special_tokens": BERT_SPECIALS}, "Template for `pair` must use both sequences"),
    ],
)
def test_template_processing_refuses_what_0_23_refuses(arguments, message):
    with pytest.raises(ValueError) as error:
        processors.TemplateProcessing(**arguments)
    assert str(error.value) == message


# The JSON 0.23 writes for each.
@pytest.mark.parametrize(
    ("key", "sequence", "written"),
    [
        (
            "post_processor",
            processors.Sequence([processors.ByteLevel(trim_offsets=False)]),
            '{"type":"Sequence","processors":[{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":false,'
            '"use_regex":true}]}',
        ),
    ],
    ids=["processors"],
)
def test_sequence_json(wiki_spec, key, sequence, written):
    assert sequence.__getstate__().decode() == written
    assert type(pickle.loads(pickle.dumps(sequence))) is type(sequence)
    tokenizer = copy.deepcopy(wiki_spec)
    setattr(tokenizer, key, sequence)
    assert type(getattr(tokenizer, key)) is type(sequence)
    compiled = tokenizer.compile()
    compiled.decode(compiled.encode(TEXT).ids)
