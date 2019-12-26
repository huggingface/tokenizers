from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    name="tokenizers",
    version="0.0.10",
    description="Fast and Customizable Tokenizers",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="NLP tokenizer BPE transformer deep learning",
    author="Anthony MOI",
    author_email="anthony@huggingface.co",
    url="https://github.com/huggingface/tokenizers",
    license="Apache License 2.0",
    rust_extensions=[RustExtension("tokenizers.tokenizers", binding=Binding.PyO3)],
    packages=["tokenizers"],
    zip_safe=False,
)
