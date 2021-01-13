echo 'prepare LDC and NIST bpe dataset'

SCRIPTS=experiment/translation/mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPEROOT=experiment/translation/subword-nmt/subword_nmt
BPE_TOKENS=60000

src=en
tgt=cn
lang=en-cn
bpe=experiment/ldc/oracle.bpe
word=experiment/ldc/oracle.word

mkdir -p $bpe $word

echo "pre-processing train data..."
# merge ldc and nist123 as training data of oracle model
for l in $src $tgt; do
    raw=experiment/ldc/ldc_raw/train.cn-en.$l
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

BPE_CODE=$bpe/code

echo "learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE


for L in $src $tgt; do
    for f in train.$L valid.$L test.$L; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $word/$f > $bpe/$f
    done
done

TEXT=experiment/ldc/oracle.bpe
fairseq-preprocess --source-lang en --target-lang cn \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/ldc.oracle.bpe.en-cn \
    --workers 20