export CUDA_VISIBLE_DEVICES="0, 1"

python train_so_2.py \
    --dataset_path "../data/rom_rea_hf_dataset" \
    --validation_split 0.1 \
    --model_name_or_path "models/literary-german-bert" \
    --output_dir "so/checkpoints/rom_rea/literary-german-bert/first" \
    --final_checkpoint_path "so/final_models/rom_rea/literary-german-bert/first" \
    --overwrite_output_dir true \
    --learning_rate 5e-5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --evaluation_strategy "steps" \
    --gradient_accumulation_steps 1 \
    --eval_steps 200 \
    --num_train_epochs 10 \
    --warmup_steps 200 \
    --logging_dir "so/logs/rom_rea/literary-german-bert/first" \
    --logging_steps 10 \
    --save_strategy "steps" \
    --save_steps 500 \
    --remove_unused_columns true \
    --logging_first_step true \
    --prediction_loss_only false \
    --seed 42 \
    "$@"