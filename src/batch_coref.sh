#! /bin/bash
dir=$1
for fname in $(ls $dir); do
    allennlp predict \
        https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz \
        $dir$fname \
        --predictor realm-coref \
        --include-package src \
        --output-file $dir$fname.coref \
        2> $dir$fname.log
done
