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

        for label in sorted(labels): #sort so we always get the same colors for a given set of labels
            colors[label] = f"hsl({h},{s}%,{l}%"
            h += h_step
        return colors

    @staticmethod
    def consecutive_chars_to_html(
        char_state_partition: List[CharState], text: str, encoding: Encoding,
    ):
        first = char_state_partition[0]
        if first.char_ix is None:
            # its a special token
            stoken = encoding.tokens[first.token_ix]
            # special tokens are represented as empty spans. We use the data attribute and css magic to display it
            return f'<span class="special-token" data-stoken={stoken}></span>'
        # We're not in a special token so this group has a start and end.
        last = char_state_partition[-1]
        start = first.char_ix
        end = last.char_ix + 1
        span_text = text[start:end]
        css_classes = []  # What css classes will we apply on the resulting span
        data_items = {}  # What data attributes will we apply on the result span
        if first.token_ix is not None:
            # We can either be in a token or not (e.g. in white space)
            css_classes.append("token")
            if first.token_ix % 2:
                # We use this to color alternating tokens.
                # A token might be split by an annotation that ends in the middle of it, so this lets us visually
                # indicate a consecutive token despite its possible splitting in the html markup
                css_classes.append("odd-token")
            else:
                # Like above, but a different color so we can see the tokens alternate
                css_classes.append("even-token")
            if encoding.tokens[first.token_ix] == "[UNK]":
                # This is a special token that is in the text. probably UNK
                css_classes.append("special-token")
                # TODO is this the right name for the data attribute ?
                data_items["stok"] = encoding.tokens[first.token_ix]
        else:
            # In this case we are looking at a group/single char that is not tokenized. e.g. white space
            css_classes.append("non-token")
        css = f'''class="{' '.join(css_classes)}"'''
        data = ""
        for key, val in data_items.items():
            data += f' data-{key}="{val}"'
        return f"<span {css} {data} >{span_text}</span>"

    @staticmethod
    def make_html(text: str, encoding: Encoding, annotations: AnnotationList):
        char_states = Aligner.__make_char_states(text, encoding, annotations)
        current_consecutive_chars = [char_states[0]]
        prev_anno_ix = char_states[0].anno_ix
        spans = []
        label_colors_dict = Aligner.calculate_label_colors(annotations)
        cur_anno_ix = char_states[0].anno_ix
        if cur_anno_ix is not None:
            # If we started in an  annotation make a span for it
            anno = annotations[cur_anno_ix]
            label = anno.label
            color = label_colors_dict[label]
            spans.append(f'<span class="annotation" style="color:{color}" data-label="{label}">')

        for cs in char_states[1:]:
            cur_anno_ix = cs.anno_ix
            if cur_anno_ix != prev_anno_ix:
                # If we've transitioned in or out of an annotation
                spans.append(
                    # Create a span from the current consecutive characters
                    Aligner.consecutive_chars_to_html(
                        current_consecutive_chars, text=text, encoding=encoding,
                    )
                )
                current_consecutive_chars = [cs]

                if prev_anno_ix is not None:
                    # if we transitioned out of an annotation close it's span
                    spans.append("</span>")
                if cur_anno_ix is not None:
                    # If we entered a new annotation make a span for it
                    anno = annotations[cur_anno_ix]
                    label = anno.label
                    color = label_colors_dict[label]
                    spans.append(
                        f'<span class="annotation" style="color:{color}" data-label="{label}">'
                    )
            prev_anno_ix = cur_anno_ix

            if cs.partition_key() == current_consecutive_chars[0].partition_key():
                # If the current charchter is in the same "group" as the previous one
                current_consecutive_chars.append(cs)
            else:
                # Otherwise we make a span for the previous group
                spans.append(
                    Aligner.consecutive_chars_to_html(
                        current_consecutive_chars, text=text, encoding=encoding,
                    )
                )
                # An reset the consecutive_char_list to form a new group
                current_consecutive_chars = [cs]
        # All that's left is to fill out the final span
        # TODO I think there is an edge case here where an annotation's span might not close
        spans.append(
            Aligner.consecutive_chars_to_html(
                current_consecutive_chars, text=text, encoding=encoding,
            )
        )
        res = HTMLBody(spans)  # Send the list of spans to the body of our html
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
    from tokenizers import Tokenizer
    from tokenizers import BertWordPieceTokenizer

    tokenizer = BertWordPieceTokenizer("/tmp/bert-base-uncased-vocab.txt", lowercase=True)

    text = """Tal 💩💩💩💩💩💩💩 💩💩 wrote this inspired by the pile of poo test which is 
    
    
    an excellent edge case test for UTF16 this lib handles well 
    
    שלום לכולם ולילה טוב
    
            💩💩💩💩💩💩💩
    
    💩💩💩💩            💩💩💩💩💩💩💩         💩💩💩
    """
    encoding = tokenizer.encode(text, add_special_tokens=True)
    print(encoding.tokens)
    anno1 = Annotation(start=0, end=2, label="foo")
    anno2 = Annotation(start=2, end=4, label="bar")
    anno3 = Annotation(start=6, end=8, label="poo")
    anno4 = Annotation(start=9, end=12, label="shoe")
    for token_ix, token in enumerate(encoding.tokens):
        token_start, token_end = encoding.token_to_chars(token_ix)
        assert token_start < len(text), f"token ${token_ix} starts after the text ends"
        assert token_end <= len(text), f"token ${token_ix} ends after the text ends"

    Aligner.make_html(
        text,
        encoding=encoding,
        annotations=[
            anno1,
            anno2,
            anno3,
            anno4,
            Annotation(start=23, end=30, label="random"),
            Annotation(start=63, end=70, label="foo"),
            Annotation(start=80, end=95, label="bar"),
            Annotation(start=120, end=128, label="bar"),
            Annotation(start=152, end=155, label="poo"),
        ],
    )
