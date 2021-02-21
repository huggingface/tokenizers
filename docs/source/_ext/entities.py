from collections import defaultdict, abc
from typing import cast

from docutils import nodes
from docutils.parsers.rst import Directive

import sphinx
from sphinx.locale import _
from sphinx.util.docutils import SphinxDirective
from sphinx.errors import ExtensionError

from conf import languages as LANGUAGES

logger = sphinx.util.logging.getLogger(__name__)

GLOBALNAME = "$GLOBAL$"


def update(d, u):
    for k, v in u.items():
        if isinstance(v, abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


class EntityNode(nodes.General, nodes.Element):
    pass


class EntitiesNode(nodes.General, nodes.Element):
    pass


class AllEntities:
    def __init__(self):
        self.entities = defaultdict(dict)

    @classmethod
    def install(cls, env):
        if not hasattr(env, "entity_all_entities"):
            entities = cls()
            env.entity_all_entities = entities
        return env.entity_all_entities

    def merge(self, other):
        self.entities.update(other.entities)

    def purge(self, docname):
        for env_docname in [GLOBALNAME, docname]:
            self.entities[env_docname] = dict(
                [
                    (name, entity)
                    for name, entity in self.entities[env_docname].items()
                    if entity["docname"] != docname
                ]
            )

    def _extract_entities(self, nodes):
        pass

    def _extract_options(self, nodes):
        pass

    def _add_entities(self, entities, language, is_global, docname):
        scope = GLOBALNAME if is_global else docname
        for entity in entities:
            name = f'{language}-{entity["name"]}'
            content = entity["content"]

            if name in self.entities[scope]:
                logger.warning(
                    f'Entity "{name}" has already been defined{" globally" if is_global else ""}',
                    location=docname,
                )

            self.entities[scope][name] = {"docname": docname, "content": content}

    def _extract_global(self, nodes):
        for node in nodes:
            if node.tagname != "field":
                raise Exception(f"Expected a field, found {node.tagname}")

            name, _ = node.children
            if name.tagname != "field_name":
                raise Exception(f"Expected a field name here, found {name_node.tagname}")

            if str(name.children[0]) == "global":
                return True

    def _extract_entities(self, nodes):
        entities = []
        for node in nodes:
            if node.tagname != "definition_list_item":
                raise Exception(f"Expected a list item here, found {node.tagname}")

            name_node, content_node = node.children
            if name_node.tagname != "term":
                raise Exception(f"Expected a term here, found {name_node.tagname}")
            if content_node.tagname != "definition":
                raise Exception(f"Expected a definition here, found {content_node.tagname}")

            name = str(name_node.children[0])
            if len(content_node.children) == 1 and content_node.children[0].tagname == "paragraph":
                content = content_node.children[0].children[0]
            else:
                content = content_node

            entities.append({"name": name, "content": content})
        return entities

    def extract(self, node, docname):
        is_global = False
        entities = []

        language = None
        for node in node.children:
            if language is None and node.tagname != "paragraph":
                raise Exception(f"Expected language name:\n.. entities:: <LANGUAGE>")
            elif language is None and node.tagname == "paragraph":
                language = str(node.children[0])
                if language not in LANGUAGES:
                    raise Exception(
                        f'Unknown language "{language}. Might be missing a newline after language"'
                    )
            elif node.tagname == "field_list":
                is_global = self._extract_global(node.children)
            elif node.tagname == "definition_list":
                entities.extend(self._extract_entities(node.children))
            else:
                raise Exception(f"Expected a list of terms/options, found {node.tagname}")

        self._add_entities(entities, language, is_global, docname)

    def resolve_pendings(self, app):
        env = app.builder.env

        updates = defaultdict(dict)
        for env_docname in self.entities.keys():
            for name, entity in self.entities[env_docname].items():
                docname = entity["docname"]
                node = entity["content"]

                for node in node.traverse(sphinx.addnodes.pending_xref):
                    contnode = cast(nodes.TextElement, node[0].deepcopy())
                    newnode = None

                    typ = node["reftype"]
                    target = node["reftarget"]
                    refdoc = node.get("refdoc", docname)
                    domain = None

                    try:
                        if "refdomain" in node and node["refdomain"]:
                            # let the domain try to resolve the reference
                            try:
                                domain = env.domains[node["refdomain"]]
                            except KeyError as exc:
                                raise NoUri(target, typ) from exc
                            newnode = domain.resolve_xref(
                                env, refdoc, app.builder, typ, target, node, contnode
                            )
                    except NoUri:
                        newnode = contnode

                    updates[env_docname][name] = {
                        "docname": docname,
                        "content": newnode or contnode,
                    }

        update(self.entities, updates)

    def get(self, language, name, docname):
        name = f"{language}-{name}"
        if name in self.entities[docname]:
            return self.entities[docname][name]
        elif name in self.entities[GLOBALNAME]:
            return self.entities[GLOBALNAME][name]
        else:
            return None


class EntitiesDirective(SphinxDirective):
    has_content = True

    def run(self):
        content = nodes.definition_list()
        self.state.nested_parse(self.content, self.content_offset, content)

        try:
            entities = AllEntities.install(self.env)
            entities.extract(content, self.env.docname)
        except Exception as err:
            raise self.error(f'Malformed directive "entities": {err}')

        return []


def entity_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    node = EntityNode()
    node.entity = text

    return [node], []


def process_entity_nodes(app, doctree, docname):
    """ Replace all the entities by their content """
    env = app.builder.env

    entities = AllEntities.install(env)
    entities.resolve_pendings(app)

    language = None
    try:
        language = next(l for l in LANGUAGES if l in app.tags)
    except Exception:
        logger.warning(f"No language tag specified, not resolving entities in {docname}")

    for node in doctree.traverse(EntityNode):
        if language is None:
            node.replace_self(nodes.Text(_(node.entity), _(node.entity)))
        else:
            entity = entities.get(language, node.entity, docname)
            if entity is None:
                node.replace_self(nodes.Text(_(node.entity), _(node.entity)))
                logger.warning(f'Entity "{node.entity}" has not been defined', location=node)
            else:
                node.replace_self(entity["content"])


def purge_entities(app, env, docname):
    """ Purge any entity that comes from the given docname """
    entities = AllEntities.install(env)
    entities.purge(docname)


def merge_entities(app, env, docnames, other):
    """ Merge multiple environment entities """
    entities = AllEntities.install(env)
    other_entities = AllEntities.install(other)
    entities.merge(other_entities)


def setup(app):
    app.add_node(EntityNode)
    app.add_node(EntitiesNode)
    app.add_directive("entities", EntitiesDirective)
    app.add_role("entity", entity_role)

    app.connect("doctree-resolved", process_entity_nodes)
    app.connect("env-merge-info", merge_entities)
    app.connect("env-purge-doc", purge_entities)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
