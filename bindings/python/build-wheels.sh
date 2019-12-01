#!/bin/bash
set -ex

curl https://sh.rustup.rs -sSf | sh -s -- --default-toolchain nightly-2019-11-01 -y
export PATH="$HOME/.cargo/bin:$PATH"
#cd /io/bindings/python

for PYBIN in /opt/python/{cp35-cp35m,cp36-cp36m,cp37-cp37m}/bin; do
    export PYTHON_SYS_EXECUTABLE="$PYBIN/python"

    "${PYBIN}/pip" install -U setuptools-rust
    "${PYBIN}/python" setup.py bdist_wheel
done

for whl in dist/*.whl; do
    auditwheel repair "$whl" -w dist/
done

# Keep only manylinux wheels
rm dist/*-linux_*

# Upload wheels
echo "Uploading to $AWS_S3_BUCKET/python/wheels"
/opt/python/cp37-cp37m/bin/pip install -U awscli
/opt/python/cp37-cp37m/bin/python -m aws s3 sync "/io/bindings/python/dist" "$AWS_S3_BUCKET/python/wheels"
