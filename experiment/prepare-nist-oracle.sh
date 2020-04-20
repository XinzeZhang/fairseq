echo 'bpeare LDC and NIST bpe dataset'

SCRIPTS=experiment/translation/mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPEROOT=experiment/translation/subword-nmt/subword_nmt
BPE_TOKENS=60000

src=en
tgt=cn
lang=en-cn
bpe=experiment/ldc/nist.bpe
word=experiment/ldc/nist.word

mkdir -p $bpe $word

echo "pre-processing train data..."
# merge nist0 and nist123 as training data of oracle model
for l in $src $tgt; do
    raw=experiment/ldc/nist_para/nist.$l.0
    nist=experiment/ldc/nist_para/nist.$l.123
    cat $raw $nist > $word/train.$l
done

echo "pre-processing valid/test data..."
for l in $src $tgt; do
    nist=experiment/ldc/nist_para/nist.$l.0
    cat $nist > $word/valid.$l
    cat $nist > $word/test.$l
done

echo "bpe train, valid, test..."
TRAIN=$word/train.en-cn
rm -f $TRAIN

for l in $src $tgt; do
    cat $word/train.$l >> $TRAIN
done

echo "loading bpe from experiment/ldc/oracle.bpe/code"
BPE_CODE=experiment/ldc/oracle.bpe/code
# python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE


for L in $src $tgt; do
    for f in train.$L valid.$L test.$L; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $word/$f > $bpe/$f
    done
done

TEXT=experiment/ldc/nist.bpe
fairseq-preprocess --source-lang en --target-lang cn \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/ldc.nist.bpe.en-cn \
    --srcdict data-bin/ldc.oracle.bpe.en-cn/dict.en.txt \
    --tgtdict data-bin/ldc.oracle.bpe.en-cn/dict.cn.txt \
    --workers 20