from string import Template
import os
from typing import List
import os

dirname = os.path.dirname(__file__)
css_filename = os.path.join(dirname, "tokenizer-styles.css")
with open(css_filename) as f:
    css = f.read()


def AnnotationTemplate(children: List[str], color: str, label: str) -> str:
    children_text = "".join(children)
    f"""<span class="annotation" style="color:{color}" data-label={label}>{children_text}</span> """


def RegularToken(text: str) -> str:
    return f"""<span class="token">{text}</span>"""


def SpecialToken(text: str, token: str) -> str:
    return f"""<span class="special-token" data-token="{token}>{text}</span>"""


def UntokenizedSpan(text: str) -> str:
    return f"""<span class="untokenized">{text}</span>"""


def Encoding(children: List[str]):
    children_text = "".join(children)
    return f"""<span class="enocding">{children_text}</span>"""


def HTMLBody(children: List[str]):
    children_text = "".join(children)
    return f"""
    <html>
        <head>
            <style>
                {css}
            </style>
        </head>
        <body>
            <div class="tokenized-text" dir=auto>
            {children_text}
            </div>
        </body>
    </html>
    """
