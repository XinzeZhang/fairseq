export CUDA_VISIBLE_DEVICES=0,1

# data_bin_dir=./data-bin/ldc.forward.bpe.join.cn-en
# model_dir=./data-bin/ldc.forward.bpe.join.cn-en
# model_name=transformer.join

data_bin_dir=./data-bin/ldc.forward.word.cn-en
model_dir=$data_bin_dir
model_name=transformer


python train.py $data_bin_dir --fp16\
    --arch transformer --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.998)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.4 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu \
    --no-epoch-checkpoints \
    --maximize-best-checkpoint-metric \
    --patience 15 \
    --source-lang cn --target-lang en --save-dir $model_dir/$model_name | tee -a $model_dir/${model_name}/training.log 
