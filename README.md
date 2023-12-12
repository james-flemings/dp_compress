# dp_compress

Command to run for fine-tuning:
```bash
python -m torch.distributed.run --nproc_per_node=8 fine_tune.py \
    --dataset wikitext \
    --subset wikitext-103-raw-v1 \
    --output_dir /data/james/models \
    --model_name gpt2-large \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 256 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --log_level info \
    --per_device_eval_batch_size 4 \
    --eval_accumulation_steps 1 \
    --seed 0 \
    --target_epsilon 6.0 \
    --per_sample_max_grad_norm 1.0 \
    --weight_decay 0.01 \
    --remove_unused_columns False \
    --num_train_epochs 20 \
    --logging_steps 5 \
    --max_grad_norm 1.0 \
    --sequence_len 512 \
    --learning_rate 0.0001 \
    --lr_scheduler_type constant \
    --dataloader_num_workers 4 \
    --disable_tqdm False \
    --load_best_model_at_end True \
    --cache_dir /data/james/.cache
```

```bash
python generate_text.py \
    --model_type gpt2-large \
    --pytorch_checkpoint /data/james/models/gpt2-large-wikitext-6.0-dp/pytorch_model.bin \
    --output_dir /data/james \
    --cache_dir /data/james/.cache \
    --dataset wikitext \
    --subset wikitext-103-raw-v1 \
    --seq_len 128 \
    --total_sequences 2 \
    --do_sample \
    --device cuda:6 \
    --prompt_len 32  
```