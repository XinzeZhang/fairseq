echo 'prepare LDC and NIST bpe dataset'

# SCRIPTS=experiment/translation/mosesdecoder/scripts
# TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
# LC=$SCRIPTS/tokenizer/lowercase.perl
# CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPEROOT=experiment/translation/subword-nmt/subword_nmt
BPE_TOKENS=60000

src=cn
tgt=en
lang=cn-en
task=forward.bpe.join
bpe=experiment/ldc/$task
word=experiment/ldc/$task/word

# mkdir -p $bpe $word

# echo "pre-processing train data..."
# # using ldc  as training data of oracle model
# for l in $src $tgt; do
#     raw=experiment/ldc/ldc_raw/train.cn-en.$l
#     cat $raw > $word/train.$l
# done

# echo "pre-processing valid data..."
# #  using nist02 as validation data of forward model
# nist02cn=experiment/ldc/ldc_raw/nist02/nist02.cn
# cat $nist02cn > $word/valid.cn
# nist02en=experiment/ldc/ldc_raw/nist02/nist02.en
# cat ${nist02en}0 > $word/valid.en
# for i in 0 1 2 3; do
#     nist02eni=${nist02en}$i
#     cat $nist02eni > $word/valid.en$i
# done

# echo "pre-processing test data..."
# #  using nist-avg as test data of forward model
# nist_avg_cn=experiment/ldc/ldc_raw/avg/avg.cn
# cat $nist_avg_cn > $word/test.cn
# nist_avg_en=experiment/ldc/ldc_raw/avg/avg.en
# cat ${nist_avg_en}0 > $word/test.en
# for i in 0 1 2 3; do
#     nist_avg_eni=${nist_avg_en}$i
#     cat $nist_avg_eni > $word/test.en$i
# done

# echo "bpe train, valid, test..."
# TRAIN=$word/train.cn-en
# rm -f $TRAIN

# for l in $src $tgt; do
#     cat $word/train.$l >> $TRAIN
# done

# BPE_CODE=$bpe/code

# echo "learn_bpe.py on ${TRAIN}..."
# python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE


# for L in $src $tgt; do
#     for f in train.$L valid.$L test.$L; do
#         echo "apply_bpe.py to ${f}..."
#         python $BPEROOT/apply_bpe.py -c $BPE_CODE < $word/$f > $bpe/$f
#     done
# done

TEXT=experiment/ldc/$task
fairseq-preprocess --source-lang cn --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/ldc.$task.cn-en \
    --workers 16 \
    --joined-dictionary

# for i in 0 1 2 3; do
#     nist_avg_eni=${nist_avg_en}$i
#     echo "apply_bpe.py to ${nist_avg_eni}..."
#     python $BPEROOT/apply_bpe.py -c $BPE_CODE < $nist_avg_eni > $bpe/test.en$i
# done