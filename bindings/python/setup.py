from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    name="tokenizers",
    version="0.0.2",
    rust_extensions=[RustExtension("tokenizers.tokenizers", binding=Binding.PyO3)],
    packages=["tokenizers"],
    zip_safe=False,
)
