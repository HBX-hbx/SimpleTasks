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

CUDA_VISIBLE_DEVICES=3

for bsz in 2048
do
    fairseq-train ${data_dir}/data-bin --arch transformer \
    --source-lang ${src} --target-lang ${tgt}  \
    --optimizer adam  --lr 0.001 --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt --max-tokens 4096  --dropout 0.3 \
    --criterion label_smoothed_cross_entropy  --label-smoothing 0.1 \
    --max-update 200000  --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --keep-last-epochs 10 --num-workers 8 \
    --save-dir ${model_dir}"/exps/checkpoints_lr_001_bsz_"${bsz} \
    --batch-size ${bsz} \
    --max-epoch 80
done