# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS     ?=
SPHINXBUILD    ?= sphinx-build
SOURCEDIR      = source
RUSTBUILDDIR   = build/rust
PYTHONBUILDDIR = build/python
NODEBUILDDIR   = build/node

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(RUSTBUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@echo "Generating doc for Rust"
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(RUSTBUILDDIR)" $(SPHINXOPTS) $(O) -t Rust
	@echo "Generating doc for Python"
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(PYTHONBUILDDIR)" $(SPHINXOPTS) $(O) -t Python
	@echo "Generating doc for Node.js"
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(NODEBUILDDIR)" $(SPHINXOPTS) $(O) -t Node
