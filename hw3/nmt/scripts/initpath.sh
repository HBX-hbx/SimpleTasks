#!/bin/sh

src="en"
tgt="zh"

ROOT="/data/private/hebingxiang"

SCRIPTS=${ROOT}"/hw3/mosesdecoder/scripts"
TOKENIZER=${SCRIPTS}"/tokenizer/tokenizer.perl"
DETOKENIZER=${SCRIPTS}"/tokenizer/detokenizer.perl"
LC=${SCRIPTS}"/tokenizer/lowercase.perl"
TRAIN_TC=${SCRIPTS}"/recaser/train-truecaser.perl"
TC=${SCRIPTS}"/recaser/truecase.perl"
DETC=${SCRIPTS}"/recaser/detruecase.perl"
NORM_PUNC=${SCRIPTS}"/tokenizer/normalize-punctuation.perl"
CLEAN=${SCRIPTS}"/training/clean-corpus-n.perl"

BPEROOT=${ROOT}"/hw3/subword-nmt"
MULTI_BLEU=${SCRIPTS}"/generic/multi-bleu.perl"
MTEVAL_V14=${SCRIPTS}"/generic/mteval-v14.pl"

data_dir=${ROOT}"/hw3/nmt/data/v16news"
model_dir=${ROOT}"/hw3/nmt/models/v16news"
utils=${ROOT}"/hw3/nmt/utils"