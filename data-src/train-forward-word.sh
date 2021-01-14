CUDA_VISIBLE_DEVICES=0 fairseq-train \
    data-bin/ldc.forward.word.cn-en \
    --arch lstm_luong_wmt_en_de --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 10}' \
    --eval-bleu-detok moses \
    # --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu \
    --no-epoch-checkpoints \
    --maximize-best-checkpoint-metric \
    --max-source-positions 80 \
    --max-target-positions 80 \
    --save-dir checkpoints/ldc-forward
    --patience 5 