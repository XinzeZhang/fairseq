echo 'prepare LDC and NIST bpe dataset'

# SCRIPTS=experiment/translation/mosesdecoder/scripts
# TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
# LC=$SCRIPTS/tokenizer/lowercase.perl
# CLEAN=$SCRIPTS/training/clean-corpus-n.perl

src=cn
tgt=en
lang=cn-en

task=forward.word.join

word=experiment/ldc/$task

mkdir -p $word

echo "pre-processing train data..."
# using ldc raw data as training data of forward model
for l in $src $tgt; do
    raw=experiment/ldc/ldc_raw/train.cn-en.$l.atok
    cat $raw > $word/train.$l
done

echo "pre-processing valid data..."
#  using nist02 as validation data of forward model
nist02cn=experiment/ldc/ldc_raw/nist02/nist02.cn
cat $nist02cn > $word/valid.cn
nist02en=experiment/ldc/ldc_raw/nist02/nist02.en
cat ${nist02en}0 > $word/valid.en
for i in 0 1 2 3; do
    nist02eni=${nist02en}$i
    cat $nist02eni > $word/valid.en$i
done

echo "pre-processing test data..."
#  using nist-avg as test data of forward model
nist_avg_cn=experiment/ldc/ldc_raw/avg/avg.cn
cat $nist_avg_cn > $word/test.cn
nist_avg_en=experiment/ldc/ldc_raw/avg/avg.en
cat ${nist_avg_en}0 > $word/test.en
for i in 0 1 2 3; do
    nist_avg_eni=${nist_avg_en}$i
    cat $nist_avg_eni > $word/test.en$i
done

fairseq-preprocess --source-lang cn --target-lang en \
    --trainpref $word/train --validpref $word/valid --testpref $word/test \
    --nwordssrc 30000 \
    --nwordstgt 30000 \
    --destdir data-bin/ldc.$task.cn-en \
    --workers 16 \
    --joined-dictionary
