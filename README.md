# dp_compress

Command to run for fine-tuning:
```bash
python -m torch.distributed.run --nproc_per_node 8 fine-tune.py \
    --dataset wikitext \
    --subset wikitext-103-raw-v1 \
    --output_dir models \
    --model_name gpt2-large \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 512 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --log_level info \
    --per_device_eval_batch_size 16 \
    --eval_accumulation_steps 1 \
    --seed 0 \
    --target_epsilon 6.0 \
    --per_sample_max_grad_norm 1.0 \
    --weight_decay 0.01 \
    --remove_unused_columns False \
    --num_train_epochs 20 \
    --logging_steps 5 \
    --max_grad_norm 1.0 \
    --sequence_len 256 \
    --learning_rate 0.0001 \
    --lr_scheduler_type constant \
    --dataloader_num_workers 8 \
    --disable_tqdm False \
    --load_best_model_at_end True 
```