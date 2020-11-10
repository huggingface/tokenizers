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
    Generates the full html with css from a list of html spans
    Args:
        children: List[str]:
            A list of strings, assumed to be html elements
        css_styles: (`optional`) str:
            Optional alternative implementation of the css
    Returns:
        An HTML string with style markup
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
