#! /bin/bash

if [ ! -d "pretrained_models_ckpts" ]; then
    echo "Downloading pretrained models..."
    make download_pretrained_models
fi

if [ -f .env ]; then
    export $(cat .env | xargs)
fi

iamcl2r -c $1
