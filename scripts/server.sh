# export CUDA_VISIBLE_DEVICES=0,1
model_name=TimesNet
ks=(1 2 3 4 5)
for k in ${ks[@]}
do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 0 \
    --model_id Server_300_300 \
    --model $model_name \
    --data Server \
    --features S \
    --seq_len 300 \
    --label_len 100\
    --root_path ./dataset/ETT-small/\
    --data_path ETTh1.csv\
    --train_epochs 20\
    --use_multi_gpu \
    --is_training 1 \
    --pred_len 300 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --d_model 16 \
    --d_ff 32 \
    --des 'Exp' \
    --itr 1 \
    --top_k $k \
    --batch_size 64 \
    --inverse
done