 MODEL_DIR=data-bin/ldc.oracle.bpe.en-cn
 fairseq-interactive \
    --path $MODEL_DIR/checkpoint_best.ldc.pt data-bin/ldc.oracle.bpe.en-cn \
    --beam 10 \
    --tokenizer moses \
    --bpe subword_nmt --bpe-codes experiment/ldc/oracle.bpe/code \
    --print-alignment --replace-unk --remove-bpe