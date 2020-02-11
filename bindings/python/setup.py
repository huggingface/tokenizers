from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    name="tokenizers",
    version="0.4.2",
    description="Fast and Customizable Tokenizers",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="NLP tokenizer BPE transformer deep learning",
    author="Anthony MOI",
    author_email="anthony@huggingface.co",
    url="https://github.com/huggingface/tokenizers",
    license="Apache License 2.0",
    rust_extensions=[RustExtension("tokenizers.tokenizers", binding=Binding.PyO3)],
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
    package_data = {
        'tokenizers': [ 'py.typed', '__init__.pyi' ],
        'tokenizers.models': [ 'py.typed', '__init__.pyi' ],
        'tokenizers.decoders': [ 'py.typed', '__init__.pyi' ],
        'tokenizers.normalizers': [ 'py.typed', '__init__.pyi' ],
        'tokenizers.pre_tokenizers': [ 'py.typed', '__init__.pyi' ],
        'tokenizers.processors': [ 'py.typed', '__init__.pyi' ],
        'tokenizers.trainers': [ 'py.typed', '__init__.pyi' ],
        'tokenizers.implementations': [ 'py.typed' ],
    },
    zip_safe=False,
)
