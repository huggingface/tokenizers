import platform
import sys
from setuptools import setup
from setuptools_rust import Binding, RustExtension


def get_py_version_cfgs():
    # For now each Cfg Py_3_X flag is interpreted as "at least 3.X"
    version = sys.version_info[0:2]
    py3_min = 5
    out_cfg = []
    for minor in range(py3_min, version[1] + 1):
        out_cfg.append("--cfg=Py_3_%d" % minor)

    if platform.python_implementation() == "PyPy":
        out_cfg.append("--cfg=PyPy")

    return out_cfg


def make_rust_extension(module_name):
    return RustExtension(
        module_name, "Cargo.toml", rustc_flags=get_py_version_cfgs(), binding=Binding.PyO3, debug=None
    )


setup(
    name="tokenizers",
    version="0.0.11",
    description="Fast and Customizable Tokenizers",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="NLP tokenizer BPE transformer deep learning",
    author="Anthony MOI",
    author_email="anthony@huggingface.co",
    url="https://github.com/huggingface/tokenizers",
    license="Apache License 2.0",
    rust_extensions=[
        make_rust_extension("tokenizers"),
        make_rust_extension("tokenizers.decoders"),
        make_rust_extension("tokenizers.encoding"),
        make_rust_extension("tokenizers.error"),
        make_rust_extension("tokenizers.models"),
        make_rust_extension("tokenizers.pretokenizers"),
        make_rust_extension("tokenizers.trainer"),
    ],
    packages=["tokenizers"],
    include_package_data=True,
    zip_safe=False,
)
