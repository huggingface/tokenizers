#!/bin/bash
set -ex

LANGUAGES=('rust' 'python' 'node')

function deploy_doc(){
    echo "Creating doc at commit $1 for language $3 and pushing to folder $2"
    git checkout $1
    cd "$GITHUB_WORKSPACE/docs"
    if [ "$2" == "master" ]; then
        echo "Pushing master"
        for LANG in "${LANGUAGES[@]}"; do
            make clean
            make html O="-t $LANG"
            ssh "$HOST_NAME" "mkdir -p $DOC_PATH/$LANG"
            scp -r build/html "$HOST_NAME:$DOC_PATH/$LANG/$2"
            cp -r build/html/_static .
        done
    elif [ "$2" == "latest" ]; then
        echo "Pushing latest for" $3
        make clean
        make html O="-t $3"
        ssh "$HOST_NAME" "mkdir -p $DOC_PATH/$3"
        scp -r build/html "$HOST_NAME:$DOC_PATH/$3/$2"
    elif ssh "$HOST_NAME" "[ -d $DOC_PATH/$3/$2 ]"; then
        echo "Directory" $2 "already exists"
        scp -r _static/* "$HOST_NAME:$DOC_PATH/$3/$2/_static/"
    else
        echo "Pushing version" $2 "for" $3
        make clean
        make html O="-t $3"
        rm -rf build/html/_static
        cp -r _static build/html
        scp -r build/html "$HOST_NAME:$DOC_PATH/$3/$2"
    fi
}

# `master` for all languages
deploy_doc "$GITHUB_SHA" master

# Rust versions
deploy_doc "$GITHUB_SHA" latest rust

# Node versions
deploy_doc "$GITHUB_SHA" latest node

# Python versions
deploy_doc "$GITHUB_SHA" v0.9.0 python
deploy_doc "$GITHUB_SHA" latest python
