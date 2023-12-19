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
    --sequence_len 128 \
    --learning_rate 0.0001 \
    --lr_scheduler_type constant \
    --dataloader_num_workers 4 \
    --disable_tqdm False \
    --load_best_model_at_end True \
    --cache_dir /data/james/.cache
```

Command for generating synthetic data:
```bash
python generate_text.py \
    --model_type gpt2-large \
    --pytorch_checkpoint /data/james/models/gpt2-large-wikitext-6.0-dp/pytorch_model.bin \
    --output_dir /data/james \
    --cache_dir /data/james/.cache \
    --dataset wikitext \
    --subset wikitext-103-raw-v1 \
    --seq_len 128 \
    --total_sequences 100000 \
    --do_sample \
    --device cuda:6 \
    --prompt_len 32  
```

Command for performing knowledge distillation:
```bash
python -m torch.distributed.run --nproc_per_node=8 knowledge_distil.py \
    --dataset wikitext \
    --subset wikitext-103-raw-v1 \
    --output_dir /data/james/models \
    --student_model distilgpt2 \
    --teacher_model gpt2-large \
    --pytorch_checkpoint /data/james/models/gpt2-large-wikitext-6.0-dp/pytorch_model.bin \
    --synthetic_data_file /data/james/synthetic_data.csv \
    --sequence_len 128 \
    --lambda_param 0.5 \
    --temperature 1.0 \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy no \
    --save_strategy no \
    --log_level info \
    --per_device_eval_batch_size 4 \
    --eval_accumulation_steps 1 \
    --seed 0 \
    --target_epsilon 3.0 \
    --per_sample_max_grad_norm 1.0 \
    --weight_decay 0.01 \
    --remove_unused_columns False \
    --num_train_epochs 10 \
    --logging_steps 5 \
    --max_grad_norm 0.0 \
    --lr_scheduler_type constant \
    --learning_rate 1e-4 \
    --dataloader_num_workers 8 \
    --disable_tqdm False \
    --load_best_model_at_end True \
    --cache_dir /data/james/.cache
```