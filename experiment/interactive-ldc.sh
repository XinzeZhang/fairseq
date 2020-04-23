 MODEL_DIR=data-bin/ldc.oracle.bpe.en-cn
 fairseq-interactive \
   $MODEL_DIR \
    --path $MODEL_DIR/checkpoint_best.ldc.pt \
    --beam 10 \
    --tokenizer moses \
    --bpe subword_nmt --bpe-codes experiment/ldc/oracle.bpe/code \
    --print-alignment --replace-unk --remove-bpe