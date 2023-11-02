model_name=TimesNet

ks=(1 2 3 4 5 10 20 30)
for k in ${ks[@]}
do
    python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model_id Server_400_100 \
        --model $model_name \
        --data Server \
        --features S \
        --seq_len 400 \
        --label_len 100\
        --root_path dataset/server/\
        --data_path server.csv\
        --train_epochs 20\
        --is_training 1 \
        --pred_len 100 \
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
        --batch_size 32 \
        --inverse \
        --freq t \
        --target values \
        --learning_rate 0.001 \
        --use_multi_gpu

done
