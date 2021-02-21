# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for those with `?=`
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
BUILDDIR      ?= build
SOURCEDIR      = source

# Put it first so that "make" without argument is like "make html_all".
html_all:
	@echo "Generating doc for Rust"
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)/rust" $(SPHINXOPTS) $(O) -t rust
	@echo "Generating doc for Python"
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)/python" $(SPHINXOPTS) $(O) -t python
	@echo "Generating doc for Node.js"
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)/node" $(SPHINXOPTS) $(O) -t node

.PHONY: html_all Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
