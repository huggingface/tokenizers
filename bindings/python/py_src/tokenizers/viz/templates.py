from string import Template
import os
from typing import List
import os

dirname = os.path.dirname(__file__)
css_filename = os.path.join(dirname, "tokenizer-styles.css")
with open(css_filename) as f:
    css = f.read()


def HTMLBody(children: List[str], css_styles=css) -> str:
    """

    :param children: A list of string rendered html nodes
    :param css_styles:  Optional css styles to replace default styles styling
    :return: An HTML string with style markup
    """
    children_text = "".join(children)
    return f"""
    <html>
        <head>
            <style>
                {css_styles}
            </style>
        </head>
        <body>
            <div class="tokenized-text" dir=auto>
            {children_text}
            </div>
        </body>
    </html>
    """
