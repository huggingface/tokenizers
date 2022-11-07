#!/bin/bash
set -ex

if ! command -v cargo &> /dev/null
then
    curl https://sh.rustup.rs -sSf | sh -s -- -y
fi

export PATH="$HOME/.cargo/bin:$PATH"

for PYBIN in /opt/python/cp{37,38,39,310,311}*/bin; do
    export PYTHON_SYS_EXECUTABLE="$PYBIN/python"

    "${PYBIN}/pip" install -U setuptools-rust setuptools wheel
    "${PYBIN}/python" setup.py bdist_wheel
    rm -rf build/*
done

for whl in ./dist/*.whl; do
    auditwheel repair "$whl" -w dist/
done

# Keep only manylinux wheels
rm ./dist/*-linux_*


# Upload wheels
/opt/python/cp37-cp37m/bin/pip install -U awscli
/opt/python/cp37-cp37m/bin/python -m awscli s3 sync --exact-timestamps ./dist "s3://tokenizers-releases/python/$DIST_DIR"
