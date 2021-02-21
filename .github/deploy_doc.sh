#!/bin/bash
set -ex

LANGUAGES=('rust' 'python' 'node')

function push_version() {
    echo "Pushing version" $2 "for" $3
    make clean
    make html O="-t $3"
    # Always use the statics from master (saved when deploying master)
    rm -rf build/html/_static
    cp -r _static build/html
    rsync -zvr --delete build/html/ "$HOST_NAME:$DOC_PATH/$3/$2"
}

function deploy_doc(){
    echo "Creating doc at $1 for language $3 and pushing to folder $2"
    git checkout $1
    cd "$GITHUB_WORKSPACE/docs"
    if [ "$2" == "master" ]; then
        echo "Pushing master"
        for LANG in "${LANGUAGES[@]}"; do
            make clean
            make html O="-t $LANG"
            ssh "$HOST_NAME" "mkdir -p $DOC_PATH/$LANG"
            rsync -zvr --delete build/html/ "$HOST_NAME:$DOC_PATH/$LANG/$2"
            cp -r build/html/_static .
        done
    elif [ "$4" != "override" ] && ssh "$HOST_NAME" "[ -d $DOC_PATH/$3/$2 ]"; then
        echo "Directory" $2 "already exists"
        rsync -zvr --delete _static/ "$HOST_NAME:$DOC_PATH/$3/$2/_static"
    else
        push_version $1 $2 $3
    fi
}

# `master` for all languages
deploy_doc "$GITHUB_SHA" master

# Rust versions
deploy_doc "$GITHUB_SHA" latest rust override

# Node versions
deploy_doc "$GITHUB_SHA" latest node override

# Python versions
deploy_doc "558f2d87795ffc9d9786f1e923398e3eebe14187" v0.9.0 python
deploy_doc "558f2d87795ffc9d9786f1e923398e3eebe14187" v0.9.1 python
deploy_doc "558f2d87795ffc9d9786f1e923398e3eebe14187" v0.9.2 python
deploy_doc "558f2d87795ffc9d9786f1e923398e3eebe14187" v0.9.3 python
deploy_doc "python-v0.9.4" v0.9.4 python
deploy_doc "python-v0.9.4" latest python override
