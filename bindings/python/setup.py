from setuptools import setup
from setuptools_rust import Binding, RustExtension

extras = {}
extras["testing"] = ["pytest"]

setup(
    name="tokenizers",
    version="0.9.4",
    description="Fast and Customizable Tokenizers",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="NLP tokenizer BPE transformer deep learning",
    author="Anthony MOI",
    author_email="anthony@huggingface.co",
    url="https://github.com/huggingface/tokenizers",
    license="Apache License 2.0",
    rust_extensions=[RustExtension("tokenizers.tokenizers", binding=Binding.PyO3, debug=False)],
    extras_require=extras,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    package_dir={"": "py_src"},
    packages=[
        "tokenizers",
        "tokenizers.models",
        "tokenizers.decoders",
        "tokenizers.normalizers",
        "tokenizers.pre_tokenizers",
        "tokenizers.processors",
        "tokenizers.trainers",
        "tokenizers.implementations",
    ],
    package_data={
        "tokenizers": ["py.typed", "__init__.pyi"],
        "tokenizers.models": ["py.typed", "__init__.pyi"],
        "tokenizers.decoders": ["py.typed", "__init__.pyi"],
        "tokenizers.normalizers": ["py.typed", "__init__.pyi"],
        "tokenizers.pre_tokenizers": ["py.typed", "__init__.pyi"],
        "tokenizers.processors": ["py.typed", "__init__.pyi"],
        "tokenizers.trainers": ["py.typed", "__init__.pyi"],
        "tokenizers.implementations": ["py.typed"],
    },
    zip_safe=False,
)
