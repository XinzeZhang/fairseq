export CUDA_VISIBLE_DEVICES=0,1
data_bin_dir=./data-bin/ldc.forward.bpe.join.cn-en
model_dir=./data-bin/ldc.forward.bpe.join.cn-en
model_name=transformer.join

python train.py $data_bin_dir --fp16\
    --arch transformer_vaswani_wmt_en_de_big --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-07 --warmup-updates 4000 --lr 0.0007 \
    --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 --update-freq 3 --log-format json --max-epoch 40 \
    --log-interval 10 --save-interval 2 --keep-last-epochs 10 \
    --seed 1111 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu \
    --no-epoch-checkpoints \
    --maximize-best-checkpoint-metric \
    --source-lang cn --target-lang en --save-dir $model_dir/$model_name | tee -a $model_dir/${model_name}/training.log 
    
#    --distributed-port 32222 --distributed-world-size 2 --ddp-backend=no_c10d

# data_bin_dir=./data-bin/ldc.forward.word.cn-en
# model_dir=./data-bin/ldc.forward.word.cn-en
# # model_name=lstm
# model_name=transformer

# python train.py $data_bin_dir \
#     --arch lstm_luong_wmt_en_de \
#     --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr-scheduler inverse_sqrt \
#     --warmup-init-lr 1e-07 --warmup-updates 4000 --lr 0.0007 --min-lr 1e-09 \
#     --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
#     --max-tokens 4096 --update-freq 3 --log-format json --max-epoch 30 \
#     --log-interval 10 --save-interval 2 --keep-last-epochs 10 \
#     --seed 1111 \
#     --eval-bleu \
#     --eval-bleu-args '{"beam": 10, "max_len_a": 1.2, "max_len_b": 20}' \
#     --eval-bleu-detok moses \
#     --eval-bleu-print-samples \
#     --best-checkpoint-metric bleu \
#     --no-epoch-checkpoints \
#     --maximize-best-checkpoint-metric \
#     --source-lang cn --target-lang en --save-dir $model_dir/$model_name | tee -a $model_dir/${model_name}/training.log 

