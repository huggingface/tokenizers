import re
from sphinx.directives.other import TocTree


class TocTreeTags(TocTree):
    hasPat = re.compile("^\s*:(.+):(.+)$")

    def filter_entries(self, entries):
        filtered = []
        for e in entries:
            m = self.hasPat.match(e)
            if m != None:
                if self.env.app.tags.has(m.groups()[0]):
                    filtered.append(m.groups()[1])
            else:
                filtered.append(e)
        return filtered

    def run(self):
        self.content = self.filter_entries(self.content)
        return super().run()


def setup(app):
    app.add_directive("toctree-tags", TocTreeTags)

    return {
        "version": "0.1",
    }
