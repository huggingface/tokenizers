from docutils import nodes

import sphinx
from sphinx.locale import _

from conf import rust_version

logger = sphinx.util.logging.getLogger(__name__)


class RustRef:
    def __call__(self, name, rawtext, text, lineno, inliner, options={}, content=[]):
        doctype = name.split(":")[1]
        parts = text.split("::")

        if text.startswith("~"):
            title = parts[-1]
            parts[0] = parts[0][1:]
        else:
            content = text
        link = self.base_link()

        if doctype == "struct":
            l, title = self.make_struct_link(parts, title)
        if doctype == "func":
            l, title = self.make_func_link(parts, title)
        if doctype == "meth":
            l, title = self.make_meth_link(parts, title)
        if doctype == "trait":
            l, title = self.make_trait_link(parts, title)
        link += l

        node = nodes.reference(internal=False, refuri=link, text=title)
        wrapper = nodes.literal(classes=["xref"])
        wrapper += node

        return [wrapper], []

    def base_link(self):
        return f"https://docs.rs/tokenizers/{rust_version}"

    def make_struct_link(self, parts, title):
        link = ""
        struct_name = parts[-1]
        path = parts[:-1]

        for p in path:
            link += f"/{p}"
        link += f"/struct.{struct_name}.html"

        return link, title

    def make_func_link(self, parts, title):
        link = ""
        fn_name = parts[-1]

        path = parts[:-1]
        for p in path:
            link += f"/{p}"
        link += f"/fn.{fn_name}.html"

        return link, title

    def make_meth_link(self, parts, title):
        meth_name = parts[-1]
        if meth_name.endswith("()"):
            meth_name = meth_name[:-2]

        link, title = self.make_struct_link(parts[:-1], title)
        link += f"#method.{meth_name}"

        if not title.endswith(")"):
            title += "()"

        return link, title

    def make_trait_link(self, parts, title):
        link = ""
        trait_name = parts[-1]

        path = parts[:-1]
        for p in path:
            link += f"/{p}"
        link += f"/trait.{trait_name}.html"

        return link, title


def setup(app):
    app.add_role("rust:struct", RustRef())
    app.add_role("rust:func", RustRef())
    app.add_role("rust:meth", RustRef())
    app.add_role("rust:trait", RustRef())

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
