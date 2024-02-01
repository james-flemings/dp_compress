# dp_compress

We are using the [dp-transformers](https://github.com/microsoft/dp-transformers) library as a submodule. To properly add it, run the following commands after cloning this repository:

```bash
git submodule init
git submodule update
```

Command to run for fine-tuning:
```bash
python -m torch.distributed.run --nproc_per_node=8 fine_tune.py \
    --data_dir dataset \
    --dataset_name yelp \
    --output_dir models \
    --model_name distilgpt2 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 128 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --log_level info \
    --per_device_eval_batch_size 64 \
    --eval_accumulation_steps 1 \
    --seed 42 \
    --target_epsilon 4.0 \
    --per_sample_max_grad_norm 1.0 \
    --weight_decay 0.01 \
    --remove_unused_columns False \
    --num_train_epochs 25 \
    --logging_steps 10 \
    --max_grad_norm 0. \
    --sequence_len 128 \
    --learning_rate 0.0001 \
    --lr_scheduler_type constant \
    --dataloader_num_workers 100 \
    --disable_tqdm False \
    --load_best_model_at_end True \
    --use_cache False \
    --cache_dir /data/james/.cache
```

Command for generating synthetic data:
```bash
python generate_text.py \
    --model_type gpt2-large \
    --pytorch_checkpoint models/gpt2-large-yelp-2.0-dp/pytorch_model.bin \
    --input_training_file dataset/train.csv \
    --output_dir dataset \
    --cache_dir /data/james/.cache \
    --dataset yelp \
    --seq_len 128 \
    --total_sequences 400000 \
    --do_sample \
    --epsilon 2.0 \
    --device cuda:0 
```

Command for performing knowledge distillation:
```bash
python -m torch.distributed.run --nproc_per_node=8 knowledge_distil.py \
    --dataset yelp \
    --output_dir models \
    --student_model distilgpt2 \
    --teacher_model gpt2-large \
    --pytorch_checkpoint models/gpt2-large-yelp-2.0-dp/pytorch_model.bin \
    --synthetic_data_file dataset/128_yelp_2.0_dp_synthetic_data.csv \
    --sequence_len 128 \
    --lambda_param 0. \
    --temperature 2.0 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --log_level info \
    --per_device_eval_batch_size 16 \
    --eval_accumulation_steps 1 \
    --seed 42 \
    --target_epsilon 2.0 \
    --per_sample_max_grad_norm 1.0 \
    --weight_decay 0.01 \
    --remove_unused_columns False \
    --num_train_epochs 1 \
    --logging_steps 50 \
    --max_grad_norm 0.0 \
    --lr_scheduler_type constant \
    --learning_rate 8e-5 \
    --dataloader_num_workers 8 \
    --disable_tqdm False \
    --load_best_model_at_end True \
    --use_cache False \
    --cache_dir /data/james/.cache
```

Command for performing differentially private knowledge distillation:
```bash
python -m torch.distributed.run --nproc_per_node=8 dp_kd.py \
    --data_dir dataset \
    --dataset_name yelp \
    --output_dir models \
    --student_model distilgpt2 \
    --teacher_model gpt2-large \
    --pytorch_checkpoint models/gpt2-large-yelp-2.0-dp/pytorch_model.bin \
    --sequence_len 128 \
    --lambda_param 0.4 \
    --temperature 2.0 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 64 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --log_level info \
    --per_device_eval_batch_size 16 \
    --eval_accumulation_steps 1 \
    --seed 42 \
    --target_epsilon 1.0 \
    --per_sample_max_grad_norm 1.0 \
    --weight_decay 0.01 \
    --remove_unused_columns False \
    --num_train_epochs 1 \
    --logging_steps 50 \
    --max_grad_norm 0.0 \
    --lr_scheduler_type constant \
    --learning_rate 1e-4 \
    --dataloader_num_workers 8 \
    --disable_tqdm False \
    --load_best_model_at_end True \
    --use_cache False \
    --cache_dir /data/james/.cache
```

Command for running performance results 

```bash
CUDA_VISIBLE_DEVICES=1 python results.py \
    --input_test_file dataset \
    --output_dir models \
    --teacher_model_type gpt2-large \
    --student_model_type distilgpt2 \
    --syn_data_teacher_file models/gpt2-large-yelp-2.0-dp \
    --syn_data_student_file models/best-distilgpt2-yelp-2.0-DPKD-syn-data \
    --dpkd_teacher_file models/gpt2-large-yelp-1.0-dp \
    --dpkd_student_file models/distilgpt2-yelp-2.0-DPKD \
    --dpsgd_student_file models/distilgpt2-yelp-2.0-dp \
    --cache_dir /data/james/.cache \
    --device cuda:0 \
    --sequence_len 128 \
    --target_epsilon 2.0
```