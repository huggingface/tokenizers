from collections import defaultdict
from typing import List, Optional, Tuple, DefaultDict, Dict
import itertools
from tokenizers import Tokenizer, Encoding
from tokenizers.viz.templates import HTMLBody
from tokenizers.viz.viztypes import (
    Aligned,
    AnnotationList,
    PartialIntList,
    CharState,
    CharStateKey,
    Annotation,
)


class Aligner:
    def __init__(self, tokenizer: Tokenizer):
        pass

    def __call__(self, text: str, encoding: Encoding, annotations: AnnotationList) -> Aligned:
        pass

    @staticmethod
    def calculate_label_colors(annotations: AnnotationList) -> Dict[str, str]:
        labels = set(map(lambda x: x.label, annotations))
        num_labels = len(labels)
        h_step = int(255 / num_labels)
        if h_step < 20:
            h_step = 20
        s = 32
        l = 64
        h = 10
        colors = {}
        for label in labels:
            colors[label] = f"hsl({h},{s}%,{l}%"
            h += h_step
        return colors

    @staticmethod
    def charstate_partition_to_html(
        char_state_partition: List[CharState],
        text: str,
        encoding: Encoding,
        annotations: AnnotationList,
    ):
        style = ""
        first = char_state_partition[0]
        if first.char_ix is None:
            # its a special token
            stoken = encoding.tokens[first.token_ix]
            return f'<span class="special-token" data-stoken={stoken}></span>'
        last = char_state_partition[-1]
        start = first.char_ix
        end = last.char_ix + 1
        span_text = text[start:end]
        css_classes = []
        data_items = {}
        if first.anno_ix is not None:
            # annotation = annotations[first.anno_ix]
            # css_classes.append("annotatixxon")
            # data_items["label"] = annotation.label
            # data_items["color"] = "blue"  # todo change this
            pass
        if first.token_ix is not None:
            css_classes.append("token")
            if first.token_ix % 2:
                css_classes.append("odd-token")
            else:
                css_classes.append("even-token")
            if encoding.special_tokens_mask[first.token_ix]:
                css_classes.append("special-token")
                data_items["stoken"] = encoding.tokens[first.token_ix]
        else:
            css_classes.append("non-token")
        css = f'''class="{' '.join(css_classes)}"'''
        data = ""
        for key, val in data_items.items():
            data += f' data-{key}="val"'
        return f"<span {css} {data} {style}>{span_text}</span>"

    @staticmethod
    def make_html(text: str, encoding: Encoding, annotations: AnnotationList):
        char_states = Aligner.__make_char_states(text, encoding, annotations)
        current_partition = [char_states[0]]
        prev_anno_ix = None
        spans = []
        label_colors_dict = Aligner.calculate_label_colors(annotations)
        for cs in char_states[1:]:
            cur_anno_ix = cs.anno_ix
            if cur_anno_ix != prev_anno_ix:
                spans.append(
                    Aligner.charstate_partition_to_html(
                        current_partition, text=text, encoding=encoding, annotations=annotations
                    )
                )
                current_partition = [cs]

                if prev_anno_ix is not None:
                    spans.append("</span>")
                if cur_anno_ix is not None:
                    anno = annotations[cur_anno_ix]
                    label = anno.label
                    color = label_colors_dict[label]
                    spans.append(
                        f'<span class="annotation" style="color:{color}" data-label="{label}">'
                    )
            prev_anno_ix = cur_anno_ix

            if cs.partition_key() == current_partition[0].partition_key():
                current_partition.append(cs)
            else:

                spans.append(
                    Aligner.charstate_partition_to_html(
                        current_partition, text=text, encoding=encoding, annotations=annotations
                    )
                )
                current_partition = [cs]
                if current_partition[0].anno_ix is not None:
                    partition_prefix = (
                        '<span class="annotation" style="color:green" data-label="xx">'
                    )
                    partition_suffix = "</span>"
                else:
                    partition_prefix = ""
                    partition_suffix = ""

        spans.append(
            Aligner.charstate_partition_to_html(
                current_partition, text=text, encoding=encoding, annotations=annotations
            )
        )
        res = HTMLBody(spans)
        with open("/tmp/out.html", "w") as f:
            f.write(res)
        return res

    @staticmethod
    def __make_anno_map(text: str, annotations: AnnotationList) -> PartialIntList:
        """

        :param text: The raw text we want to align to
        :param annotations: A (possibly empty) list of annotations
        :return: A list of  length len(text) whose entry at index i is None if there is no annotation on charachter i
            or k, the index of the annotation that covers index i where k is with respect to the list of annotations
        """
        annotation_map = [None] * len(text)
        for anno_ix, a in enumerate(annotations):
            for i in range(a.start, a.end):
                annotation_map[i] = anno_ix
        return annotation_map

    @staticmethod
    def __make_token_and_word_map(
        text: str, encoding: Encoding
    ) -> Tuple[PartialIntList, PartialIntList]:
        """

        :param text: The text being aligned
        :param encoding: The encoding of the text returned by a tokenizer
        :return: A tuple of lists, each list is of length len(txt). The first list maps a charachter to a word
         and the second list maps a charachter to a token. At index i, a value is None if there is no word/token that
         corresponds to that charchter, otherwise the value is the index of the word/charachter in the encodings respective list
        """
        word_map: PartialIntList = [None] * len(text)
        token_map: PartialIntList = [None] * len(text)
        for token_ix, word_ix in enumerate(encoding.words):
            token_start, token_end = encoding.token_to_chars(token_ix)
            word_start, word_end = encoding.word_to_chars(word_ix) if word_ix else (0, 0)
            for char_ix in range(token_start, token_end):
                token_map[char_ix] = token_ix
            for char_ix in range(word_start, word_end):
                word_map[char_ix] = word_ix
        return word_map, token_map

    @staticmethod
    def __make_char_states(
        text: str, encoding: Encoding, annotations: AnnotationList
    ) -> List[CharState]:
        """
        For each charachter in the original text, we emit a tuple representing it's "state" -
        which token_ix it corresponds to
        which word_ix it corresponds to
        which annotation_ix it corresponds to

        :param text:
        :param encoding:
        :param annotations:
        :return:
        """
        annotation_map = Aligner.__make_anno_map(text, annotations)
        word_map, token_map = Aligner.__make_token_and_word_map(text, encoding)
        # Todo make this a dataclass or named tuple
        char_states: List[CharState] = [
            CharState(char_ix=char_ix, word_ix=word_ix, token_ix=token_ix, anno_ix=anno_ix)
            for char_ix, (word_ix, token_ix, anno_ix) in enumerate(
                itertools.zip_longest(word_map, token_map, annotation_map, fillvalue=None)
            )
        ]
        return char_states


if __name__ == "__main__":
    from tokenizers.trainers import BpeTrainer
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    import tokenizers

    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()

    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.train(trainer, ["/home/tal/dev/lighttag/sample_data/fedreg.json"])
    text = "i am tal  and I wrote this 💩💩💩 xxx"
    encoding = tokenizer.encode(text, add_special_tokens=True)
    anno1 = Annotation(start=3, end=6, label="foo")
    anno2 = Annotation(start=8, end=7, label="bar")
    anno3 = Annotation(start=12, end=15, label="zab")
    for token_ix, token in enumerate(encoding.tokens):
        token_start, token_end = encoding.token_to_chars(token_ix)
        assert token_start < len(text), f"token ${token_ix} starts after the text ends"
        assert token_end <= len(text), f"token ${token_ix} ends after the text ends"

    Aligner.make_html(text, encoding=encoding, annotations=[anno1, anno2, anno3])
