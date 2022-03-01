#! /bin/bash

MACOSX_DEPLOYMENT_TARGET=10.11 SDKROOT="/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk" CFLAGS="-I/usr/include/openssl -I/usr/local/opt/readline/include -I/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include" CPPFLAGS="-I/usr/local/opt/zlib/include" LDFLAGS="-L/usr/lib -L/usr/local/opt/readline/lib" pyenv install <wtv_version>

~/.pyenv/versions/<wtv_version>/bin/pip install setuptools wheel setuptools-rust==0.11.3 --ignore-installed --force-reinstall

MACOSX_DEPLOYMENT_TARGET=10.11 ~/.pyenv/versions/<wtv_version>/bin/python setup.py bdist_wheel

