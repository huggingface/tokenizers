🤗 Tokenizers is tested on Python 3.5+.

You should install 🤗 Tokenizers in a
`virtual environment <https://docs.python.org/3/library/venv.html>`_. If you're unfamiliar with
Python virtual environments, check out the
`user guide <https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/>`__.
Create a virtual environment with the version of Python you're going to use and activate it.

Installation with pip
----------------------------------------------------------------------------------------------------

🤗 Tokenizers can be installed using pip as follows::

    pip install tokenizers


Installation from sources
----------------------------------------------------------------------------------------------------

To use this method, you need to have the Rust language installed. You can follow
`the official guide <https://www.rust-lang.org/learn/get-started>`__ for more information.

If you are using a unix based OS, the installation should be as simple as running::

    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

Or you can easiy update it with the following command::

    rustup update

Once rust is installed, we can start retrieving the sources for 🤗 Tokenizers::

    git clone https://github.com/huggingface/tokenizers

Then we go into the python bindings folder::

    cd tokenizers/bindings/python

At this point you should have your `virtual environment`_ already activated. In order to
compile 🤗 Tokenizers, you need to install the Python package :obj:`setuptools_rust`::

    pip install setuptools_rust

Then you can have 🤗 Tokenizers compiled and installed in your virtual environment with
the following command::

    python setup.py install
