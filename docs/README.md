## Requirements

In order to generate the documentation, it is necessary to have a Python environment with the
following:
```python
pip install sphinx sphinx_rtd_theme setuptools_rust
```

It is also necessary to have the `tokenizers` library in this same environment, for Sphinx to
generate all the API Reference and links properly.  If you want to visualize the documentation with
some modifications made to the Python bindings, make sure you build it from source.

## Building the documentation

Once everything is setup, you can build the documentation automatically for all the languages
using the following command in the `/docs` folder:

```bash
make html_all
```

If you want to build only for a specific language, you can use:

```bash
make html O="-t python"
```

(Replacing `python` by the target language among `rust`, `node`, and `python`)


**NOTE**

If you are making any structural change to the documentation, it is recommended to clean the build
directory before rebuilding:

```bash
make clean && make html_all
```
