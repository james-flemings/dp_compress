import torch
import transformers
import datasets
import math
import os
import argparse
from tqdm import tqdm

def main(args):
    teacher_model_6_dp = transformers.GPT2LMHeadModel.from_pretrained(args.teacher_model_type, cache_dir=args.cache_dir)
    teacher_pre_trained_model = transformers.GPT2LMHeadModel.from_pretrained(args.teacher_model_type, cache_dir=args.cache_dir) 
    #student_model_syn = transformers.GPT2LMHeadModel.from_pretrained("/data/james/models/distilgpt2-DPKD-syn-data", local_files_only=True)
    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.teacher_model_type, cache_dir=args.cache_dir)
    num_added_toks = tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    mean_tok_emb = teacher_pre_trained_model.transformer.wte.weight.data.mean(dim=0)
    # Initialize the newly-added token embedding to the mean of all token embeddings
    for i in range(num_added_toks):
        teacher_pre_trained_model.transformer.wte.weight.data[-(i + 1), :] = mean_tok_emb
        teacher_model_6_dp.transformer.wte.weight.data[-(i + 1), :] = mean_tok_emb

    teacher_model_6_dp.resize_token_embeddings(len(tokenizer))
    teacher_pre_trained_model.resize_token_embeddings(len(tokenizer))

    sd = torch.load(os.path.join(args.syn_data_teacher_file, "pytorch_model.bin"), map_location="cpu")
    state_dict = {}
    for key, value in sd.items():
        key = key.replace("_module.module.", "")
        state_dict[key] = value

    teacher_model_6_dp.load_state_dict(state_dict)
    #teacher_model_6_dp.tie_weights()
    teacher_model_6_dp = teacher_model_6_dp.to("cuda:0")
    #student_model_syn = student_model_syn.to("cuda:0")
    teacher_pre_trained_model = teacher_pre_trained_model.to('cuda:0')

    # Load dataset
    data_path_test = os.path.join(args.input_test_file, "test.csv")
    dataset = datasets.load_dataset('csv', data_files={'test': data_path_test}, cache_dir=args.cache_dir)

    label_column_names = [name for name in dataset["test"].column_names if "label" in name]
    # Tokenize data
    block_size = args.sequence_len

    def preprocess_function(examples):
        return tokenizer(examples['text'])

    def group_function(examples):
        batch = []
        for t in range(len(examples['text'])):
            text = "\t".join([examples[name][t] for name in label_column_names]) + "\n\n" + examples['text'][t] + tokenizer.eos_token
            batch.append(text)

        result = tokenizer(batch, padding="max_length", truncation=True,
                           max_length=args.sequence_len)
        concatenated_examples = {k: sum(result[k], []) for k in result.keys()}
        total_length = len(concatenated_examples[list(result.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Tokenize data
    dataset = dataset.map(preprocess_function, remove_columns=dataset.column_names['test'])
    dataset = dataset['test'].map(
        group_function, batched=True, desc="tokenizing dataset",
        num_proc=2
    )
    train_args = transformers.TrainingArguments(output_dir=args.output_dir, per_device_eval_batch_size=8, label_names=['labels'])
    trainer_teacher_6_dp = transformers.Trainer(model=teacher_model_6_dp, args=train_args)
    trainer_teacher_pre = transformers.Trainer(model=teacher_pre_trained_model, args=train_args)
    #trainer_student_syn = transformers.Trainer(model=student_model_syn, args=train_args)
    print(f"Test set perplexity of pre-trained Teacher model \
          {math.exp(trainer_teacher_pre.evaluate(eval_dataset=dataset)['eval_loss']):.2f}")
    print(f"Test set perplexity of Teacher model with DP-SGD ε = 6 \
           {math.exp(trainer_teacher_6_dp.evaluate(eval_dataset=dataset)['eval_loss']):.2f}")
    #print(f"Test set perplexity of Student model trained with synthetic data \
    #      {math.exp(trainer_student_syn.evaluate(eval_dataset=lm_dataset)['eval_loss']):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_test_file",
        default=None,
        type=str,
        required=True
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True
    )
    parser.add_argument(
        "--teacher_model_type",
        type=str,
        default=None,
        required=True
    )
    parser.add_argument(
        "--student_model_type",
        type=str,
        default=None,
        required=True
    )
    parser.add_argument(
        "--syn_data_teacher_file",
        type=str,
        default=None,
        required=True
    )
    parser.add_argument(
        "--syn_data_student_file",
        type=str,
        default=None,
        required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        required=False
    )
    parser.add_argument(
        "--sequence_len",
        type=int,
        default=None,
        required=True
    )
    args = parser.parse_args()
    main(args)