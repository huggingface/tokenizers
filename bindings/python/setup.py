from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    name="tokenizers",
    version="0.0.2",
    description="Fast and Customizable Tokenizers",
    author="Anthony MOI",
    author_email="anthony@huggingface.co",
    url="https://github.com/huggingface/tokenizers",
    license="Apache License 2.0",
    rust_extensions=[RustExtension("tokenizers.tokenizers", binding=Binding.PyO3)],
    packages=["tokenizers"],
    zip_safe=False,
)
