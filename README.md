# Differentially Private Knowledge Distillation via Synthetic Text Generation

This is the source code for the paper [Differentially Private Knowledge Distillation via Synthetic Text Generation](https://arxiv.org/pdf/2403.00932). In this work, we improved differentially private knowledge distillation of generative large language models by expoiting differentially prviate synthetic data for training the student. Our framework proceeds in three steps: (1) A teaher model is fine-tuned on a private downstream dataset using DP-SGD. (2) The teacher model is prompt with control codes to generate synthetic data. (3) A student model is trained on the synthetic data with knowledge distillation from the teacher.

## Environment Setup
We are using the [dp-transformers](https://github.com/microsoft/dp-transformers) library as a submodule. To properly add it, run the following commands after cloning this repository:

```bash
git submodule init
git submodule update
```

Command to run for fine-tuning:
```bash
python -m torch.distributed.run --nproc_per_node=8 dp_fine_tune.py \
    --data_dir /data/james/big_patent_data \
    --dataset_name big_patent \
    --output_dir /data/james/models \
    --model_name distilgpt2 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --log_level info \
    --per_device_eval_batch_size 32 \
    --eval_accumulation_steps 1 \
    --seed 42 \
    --target_epsilon 2.0 \
    --per_sample_max_grad_norm 1.0 \
    --weight_decay 0.01 \
    --remove_unused_columns False \
    --num_train_epochs 30 \
    --logging_steps 5 \
    --max_grad_norm 0. \
    --sequence_len 128 \
    --learning_rate 0.0001 \
    --lr_scheduler_type constant \
    --dataloader_num_workers 10 \
    --disable_tqdm False \
    --load_best_model_at_end True \
    --use_cache True \
    --use_cc False \
    --cache_dir /data/james/.cache
```

Command for generating synthetic data:
```bash
python generate_text.py \
    --model_type gpt2-large \
    --pytorch_checkpoint /data/james/models/cc-gpt2-large-big_patent-2.0-dp/pytorch_model.bin \
    --input_training_file /data/james/big_patent_data/train.csv \
    --output_dir /data/james/synthetic_data \
    --use_cache True \
    --cache_dir /data/james/.cache \
    --dataset big_patent \
    --seq_len 128 \
    --batch_size 64 \
    --total_sequences 600000 \
    --do_sample \
    --epsilon 2.0 \
    --device cuda:7 
```

Command for performing knowledge distillation:
```bash
python -m torch.distributed.run --nproc_per_node=8 knowledge_distil.py \
    --dataset big_patent \
    --output_dir /data/james/models \
    --student_model distilgpt2 \
    --teacher_model gpt2-large \
    --pytorch_checkpoint /data/james/models/cc-gpt2-large-big_patent-2.0-dp/pytorch_model.bin \
    --synthetic_data_file /data/james/synthetic_data/128_big_patent_2.0_dp_synthetic_data.csv \
    --sequence_len 128 \
    --lambda_param 0.4 \
    --alpha_cos 0 \
    --temperature 1.0 \
    --per_device_train_batch_size 64 \
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
    --num_train_epochs 14\
    --logging_steps 50 \
    --max_grad_norm 0.0 \
    --warmup_step 0 \
    --lr_scheduler_type constant \
    --learning_rate 8e-5 \
    --dataloader_num_workers 8 \
    --disable_tqdm False \
    --load_best_model_at_end True \
    --use_cache True \
    --cache_dir /data/james/.cache
```

Command for performing differentially private knowledge distillation:
```bash
python -m torch.distributed.run --nproc_per_node=8 dp_kd.py \
    --data_dir /data/james/big_patent_data \
    --dataset_name big_patent \
    --output_dir /data/james/models \
    --student_model distilgpt2 \
    --teacher_model gpt2-large \
    --pytorch_checkpoint /data/james/models/gpt2-large-big_patent-1.0-dp/pytorch_model.bin \
    --sequence_len 128 \
    --lambda_param 0.4 \
    --temperature 1.0 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 16 \
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
    --num_train_epochs 30 \
    --logging_steps 50 \
    --max_grad_norm 0.0 \
    --lr_scheduler_type constant \
    --learning_rate 1e-4 \
    --dataloader_num_workers 8 \
    --disable_tqdm False \
    --load_best_model_at_end True \
    --use_cache True \
    --cache_dir /data/james/.cache
```

Command for running performance results for yelp:
```bash
python results.py \
    --input_test_file /data/james/yelp_data \
    --output_dir /data/james/models \
    --teacher_model_type gpt2-large \
    --student_model_type distilgpt2 \
    --syn_data_teacher_file /data/james/models/gpt2-large-yelp-2.0-dp \
    --syn_data_student_file /data/james/models/distilgpt2-yelp-2.0-dp-syn-data \
    --dpkd_syn_data_student_file /data/james/models/best-distilgpt2-yelp-2.0-DPKD-syn-data \
    --dpkd_teacher_file /data/james/models/gpt2-large-yelp-1.0-dp \
    --dpkd_student_file /data/james/models/distilgpt2-yelp-2.0-DPKD \
    --dpsgd_student_file /data/james/models/distilgpt2-yelp-2.0-dp \
    --use_cache True \
    --cache_dir /data/james/.cache \
    --device cuda:0 \
    --sequence_len 128 \
    --target_epsilon 2.0
```

Command for running performance results for big patent:
```bash
python results.py \
    --input_test_file /data/james/big_patent \
    --output_dir /data/james/models \
    --teacher_model_type gpt2-large \
    --student_model_type distilgpt2 \
    --syn_data_teacher_file /data/james/models/cc-gpt2-large-big_patent-2.0-dp \
    --syn_data_student_file /data/james/models/distilgpt2-big_patent-2.0-dp-syn-data \
    --dpkd_syn_data_student_file /data/james/models/best-distilgpt2-big_patent-2.0-DPKD-syn-data \
    --dpkd_teacher_file /data/james/models/gpt2-large-big_patent-1.0-dp \
    --dpkd_student_file /data/james/models/distilgpt2-big_patent-2.0-DPKD \
    --dpsgd_student_file /data/james/models/distilgpt2-big_patent-2.0-dp \
    --use_cache True \
    --cache_dir /data/james/.cache \
    --device cuda:0 \
    --sequence_len 128 \
    --target_epsilon 2.0
```

Command for running performance results for dbpedia_14:
```bash
python results.py \
    --input_test_file /data/james/dbpedia_14_data \
    --output_dir /data/james/models \
    --teacher_model_type gpt2-large \
    --student_model_type distilgpt2 \
    --syn_data_teacher_file /data/james/models/cc-gpt2-large-dbpedia_14-2.0-dp \
    --syn_data_student_file /data/james/models/distilgpt2-dbpedia_14-2.0-dp-syn-data \
    --dpkd_syn_data_student_file /data/james/models/best-distilgpt2-dbpedia_14-2.0-DPKD-syn-data \
    --dpkd_teacher_file /data/james/models/gpt2-large-dbpedia_14-1.0-dp \
    --dpkd_student_file /data/james/models/distilgpt2-dbpedia_14-2.0-DPKD \
    --dpsgd_student_file /data/james/models/distilgpt2-dbpedia_14-2.0-dp \
    --use_cache True \
    --cache_dir /data/james/.cache \
    --device cuda:0 \
    --sequence_len 128 \
    --target_epsilon 2.0
```

## Citation
If you found this repository useful, please consider citing our work:
```stex
@article{flemings2024differentially,
  title={Differentially private knowledge distillation via synthetic text generation},
  author={Flemings, James and Annavaram, Murali},
  journal={arXiv preprint arXiv:2403.00932},
  year={2024}
}
```
