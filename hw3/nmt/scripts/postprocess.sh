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

CUDA_VISIBLE_DEVICES=4

model_list=`ls ${model_dir}/exps`

for model in ${model_list}
do
    dir=`echo $model | cut -d'_' -f 2,3,4,5`
    grep ^H ${data_dir}/result/${dir}/bestbeam8.txt | cut -f3- > ${data_dir}/result/${dir}/predict.seg.tok.bpe.zh
    grep ^T ${data_dir}/result/${dir}/bestbeam8.txt | cut -f2- > ${data_dir}/result/${dir}/answer.seg.tok.bpe.zh

	sed -r 's/(@@ )| (@@ ?$)//g' < ${data_dir}/result/${dir}/predict.seg.tok.bpe.zh  > ${data_dir}/result/${dir}/predict.seg.tok.zh
	sed -r 's/(@@ )| (@@ ?$)//g' < ${data_dir}/result/${dir}/answer.seg.tok.bpe.zh  > ${data_dir}/result/${dir}/answer.seg.tok.zh

	${MULTI_BLEU} -lc ${data_dir}/result/${dir}/answer.seg.tok.zh < ${data_dir}/result/${dir}/predict.seg.tok.zh > ${data_dir}/result/${dir}/bleu.txt

	${DETOKENIZER} -l zh < ${data_dir}/result/${dir}/predict.seg.tok.zh > ${data_dir}/result/${dir}/predict.seg.zh
done