#!/bin/bash
set -ex

curl https://sh.rustup.rs -sSf | sh -s -- -y
export PATH="$HOME/.cargo/bin:$PATH"

for PYBIN in /opt/python/{cp37-cp37m,cp38-cp38,cp39-cp39,cp310-cp310}/bin; do
    export PYTHON_SYS_EXECUTABLE="$PYBIN/python"

    "${PYBIN}/pip" install -U setuptools-rust==0.11.3
    # --static-embedding allows the use of a static python interpreter embedding
    # which is necessary for quay manylinux builds.
    "${PYBIN}/python" setup.py bdist_wheel --static-embedding
    rm -rf build/*
done

for whl in dist/*.whl; do
    auditwheel repair "$whl" -w dist/
done

# Keep only manylinux wheels
rm dist/*-linux_*

# Upload wheels
/opt/python/cp37-cp37m/bin/pip install -U awscli
/opt/python/cp37-cp37m/bin/python -m awscli s3 sync --exact-timestamps ./dist "s3://tokenizers-releases/python/$DIST_DIR"
