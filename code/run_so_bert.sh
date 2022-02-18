export CUDA_VISIBLE_DEVICES="0,1"

python train_so_2.py \
    --dataset_path "../data/rom_rea_hf_dataset" \
    --window 512 \
    --padding 160 \
    --stride 32 \
    --normalize_targets true \
    --validation_split 0.1 \
    --model_name_or_path "uklfr/gottbert-base" \
    --output_dir "so/checkpoints/rom_rea/gottbert-base/second" \
    --final_checkpoint_path "so/final_models/rom_rea/gottbert-base/second" \
    --overwrite_output_dir true \
    --learning_rate 5e-5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --evaluation_strategy "steps" \
    --gradient_accumulation_steps 1 \
    --eval_steps 500 \
    --num_train_epochs 7 \
    --warmup_steps 200 \
    --logging_dir "so/logs/rom_rea/gottbert-base/second" \
    --logging_steps 10 \
    --save_strategy "steps" \
    --save_steps 2500 \
    --save_total_limit 20 \
    --remove_unused_columns true \
    --logging_first_step true \
    --prediction_loss_only false \
    --seed 42 \
    "$@"