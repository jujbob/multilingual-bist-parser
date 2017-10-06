#!/bin/bash

PRJ_DIR=/home/ktlim/parser/multilingual-bist-parser
CORPUS_DIR=${PRJ_DIR}/corpus/ud-treebanks-v2.0
OUT_DIR=./
EXT_DIR=${PRJ_DIR}/external_embeddings
python=/usr/bin/python


    mkdir -p ${OUT_DIR}/results_sammi/
    ${python} ${PRJ_DIR}/bist-parser/bmstparser/src/parser.py \
                --dynet-seed 123456789 \
                --outdir ${OUT_DIR}/results_sammi/ \
                --train ${PRJ_DIR}/training_file_list.txt \
                --dev ${CORPUS_DIR}/UD_North_Sami/sme-ud-test.conllu \
                --epochs 1 \
                --dynet-mem 3072 \
                --lstmdims 125 \
                --lstmlayers 2 \
                --bibi-lstm \
                --train_multilingual \
                --multilingual_emb \
                --extConcateFlag \
                --extcluster ${EXT_DIR}/model_sme_fin_no.vec \
                --train_lang North_Sami \
                --test_lang sme


#                --dev ${CORPUS_DIR}/UD_English/en-ud-dev.conllu \
#               --train_multilingual \
#               --multilingual_emb \
#               --extConcateFlag \
#                --add_lang_vec \
#                --extr /home/ktlim/git/bist-parser/External_Embedding/average.100++.filtered \
#                --extcluster /home/ktlim/git/bist-parser/External_Embedding/multiCluster.100.filtered \
