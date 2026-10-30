import copy
import json
import pickle

import pytest

from tokenizers import processors

PAIR = ("Hello world", "how are you?")
A = {"seq": "A", "type_id": 0}
B1 = {"seq": "B", "type_id": 1}
CLS = {"ids": [1], "type_id": 0}
SEP = {"ids": [2], "type_id": 0}
SEP1 = {"ids": [2], "type_id": 1}


@pytest.mark.parametrize(
    ("post_processor", "written"),
    [
        (processors.TemplateProcessing(), {"single": [A], "pair": [A, B1]}),
        (
            processors.TemplateProcessing(single=[CLS, A, SEP], pair=[CLS, A, SEP, B1, SEP1]),
            {"single": [CLS, A, SEP], "pair": [CLS, A, SEP, B1, SEP1]},
        ),
        (
            processors.TemplateProcessing(single=[{"ids": [1]}, {"seq": "A"}], pair=[{"seq": "A"}, B1]),
            {"single": [CLS, A], "pair": [A, B1]},
        ),
    ],
)
def test_template_processing_json(bert_spec, post_processor, written):
    written = {"type": "TemplateProcessing", **written}
    assert json.loads(post_processor.__getstate__()) == written
    twin = pickle.loads(pickle.dumps(post_processor))
    assert type(twin) is processors.TemplateProcessing
    bert_spec.post_processor = post_processor
    assert type(bert_spec.post_processor) is processors.TemplateProcessing
    assert json.loads(bert_spec.to_str())["post_processor"] == written
    compiled = bert_spec.compile()
    compiled.decode(compiled.encode(PAIR).ids)


# transformers rebuilds the template from the tokenizer's own bos/eos ids.
def test_template_processing_encodes_like_the_file_it_was_rebuilt_from(bert, bert_spec):
    tokenizer = copy.deepcopy(bert_spec)
    tokenizer.post_processor = processors.TemplateProcessing(single=[CLS, A, SEP], pair=[CLS, A, SEP, B1, SEP1])
    rebuilt, declared = tokenizer.compile().encode(PAIR), bert.encode(PAIR)
    assert (rebuilt.ids, rebuilt.type_ids) == (declared.ids, declared.type_ids)


def test_template_processing_refuses_a_piece_of_neither_shape():
    with pytest.raises(TypeError):
        processors.TemplateProcessing(single=[{"tokens": ["[CLS]"]}])
