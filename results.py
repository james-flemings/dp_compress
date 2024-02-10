import torch
import transformers
import datasets
import math
import os
import argparse
from tqdm import tqdm

def main(args):
    if args.use_cache:
        teacher_model_dpkd = transformers.GPT2LMHeadModel.from_pretrained(args.teacher_model_type, cache_dir=args.cache_dir)
        student_model_dpkd = transformers.GPT2LMHeadModel.from_pretrained(args.student_model_type, cache_dir=args.cache_dir)
        student_model_dpsgd = transformers.GPT2LMHeadModel.from_pretrained(args.student_model_type, cache_dir=args.cache_dir)
        teacher_pre_trained_model = transformers.GPT2LMHeadModel.from_pretrained(args.teacher_model_type, cache_dir=args.cache_dir) 
        student_pre_trained_model = transformers.GPT2LMHeadModel.from_pretrained(args.student_model_type, cache_dir=args.cache_dir) 
    else:
        teacher_model_dpkd = transformers.GPT2LMHeadModel.from_pretrained(args.teacher_model_type)
        student_model_dpkd = transformers.GPT2LMHeadModel.from_pretrained(args.student_model_type)
        student_model_dpsgd = transformers.GPT2LMHeadModel.from_pretrained(args.student_model_type)
        teacher_pre_trained_model = transformers.GPT2LMHeadModel.from_pretrained(args.teacher_model_type) 
        student_pre_trained_model = transformers.GPT2LMHeadModel.from_pretrained(args.student_model_type) 
    student_model_syn = transformers.GPT2LMHeadModel.from_pretrained(args.syn_data_student_file, local_files_only=True)
        
    # Load tokenizer
    if args.use_cache:
        teacher_tokenizer = transformers.AutoTokenizer.from_pretrained(args.teacher_model_type, cache_dir=args.cache_dir)
        student_tokenizer = transformers.AutoTokenizer.from_pretrained(args.student_model_type, cache_dir=args.cache_dir)
    else:
        teacher_tokenizer = transformers.AutoTokenizer.from_pretrained(args.teacher_model_type)
        student_tokenizer = transformers.AutoTokenizer.from_pretrained(args.student_model_type)

    num_added_toks = teacher_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    num_added_toks = student_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    teacher_mean_tok_emb = teacher_pre_trained_model.transformer.wte.weight.data.mean(dim=0)
    student_mean_tok_emb = student_pre_trained_model.transformer.wte.weight.data.mean(dim=0)
    # Initialize the newly-added token embedding to the mean of all token embeddings
    for i in range(num_added_toks):
        teacher_pre_trained_model.transformer.wte.weight.data[-(i + 1), :] = teacher_mean_tok_emb
        #student_pre_trained_model.transformer.wte.weight.data[-(i + 1), :] = student_mean_tok_emb
        teacher_model_dpkd.transformer.wte.weight.data[-(i + 1), :] = teacher_mean_tok_emb
        student_model_dpkd.transformer.wte.weight.data[-(i + 1), :] = student_mean_tok_emb
        student_model_syn.transformer.wte.weight.data[-(i + 1), :] = student_mean_tok_emb
        student_model_dpsgd.transformer.wte.weight.data[-(i + 1), :] = student_mean_tok_emb

    teacher_model_dpkd.resize_token_embeddings(len(teacher_tokenizer))
    student_model_dpkd.resize_token_embeddings(len(student_tokenizer))
    teacher_pre_trained_model.resize_token_embeddings(len(teacher_tokenizer))
    #student_pre_trained_model.resize_token_embeddings(len(student_tokenizer))
    student_model_syn.resize_token_embeddings(len(teacher_tokenizer))
    student_model_dpsgd.resize_token_embeddings(len(student_tokenizer))

    # Load dp weights for teacher dpkd
    sd = torch.load(os.path.join(args.dpkd_teacher_file, "pytorch_model.bin"), map_location="cpu")
    state_dict = {}
    for key, value in sd.items():
        key = key.replace("_module.module.", "")
        state_dict[key] = value
    teacher_model_dpkd.load_state_dict(state_dict)

    # Load dp weights for student dpkd
    sd = torch.load(os.path.join(args.dpkd_student_file, "pytorch_model.bin"), map_location="cpu")
    state_dict = {}
    for key, value in sd.items():
        key = key.replace("_module.module.", "")
        state_dict[key] = value
    student_model_dpkd.load_state_dict(state_dict)

    # Load dp weights for student  
    sd = torch.load(os.path.join(args.dpsgd_student_file, "pytorch_model.bin"), map_location="cpu")
    state_dict = {}
    for key, value in sd.items():
        key = key.replace("_module.module.", "")
        state_dict[key] = value
    student_model_dpsgd.load_state_dict(state_dict)

    teacher_model_dpkd = teacher_model_dpkd.to(args.device)
    student_model_dpkd = student_model_dpkd.to(args.device)
    student_model_syn = student_model_syn.to(args.device)
    student_model_dpsgd = student_model_dpsgd.to(args.device)
    teacher_pre_trained_model = teacher_pre_trained_model.to(args.device)
    student_pre_trained_model = student_pre_trained_model.to(args.device)

    # Load dataset
    data_path_test = os.path.join(args.input_test_file, "test.csv")
    if args.use_cache:
        dataset = datasets.load_dataset('csv', data_files={'test': data_path_test}, cache_dir=args.cache_dir)
    else:
        dataset = datasets.load_dataset('csv', data_files={'test': data_path_test})

    label_column_names = [name for name in dataset["test"].column_names if "label" in name]
    # Tokenize data
    block_size = args.sequence_len

    def group_function(examples, tokenizer):
        batch = examples
        #batch = []
        #for t in range(len(examples['text'])):
        #    text = "\t".join([examples[name][t] for name in label_column_names]) + "\n\n" + examples['text'][t] + tokenizer.eos_token
        #    batch.append(text)

        #result = tokenizer(batch, padding="max_length", truncation=True,
        #                   max_length=args.sequence_len)
        result = tokenizer(batch['text'])
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
    teacher_dataset = dataset['test'].map(
        group_function, fn_kwargs={'tokenizer': teacher_tokenizer}, 
        batched=True, desc="tokenizing dataset",
        remove_columns=dataset.column_names['test'],
    )
    student_dataset = dataset['test'].map(
        group_function, fn_kwargs={'tokenizer': student_tokenizer}, 
        batched=True, desc="tokenizing dataset",
        remove_columns=dataset.column_names['test']
    )
    '''
    calculate PPL of fixed length model
    Code obtained from: https://huggingface.co/docs/transformers/en/perplexity
    '''

    nlls_pre_trained = []
    nlls_dpsgd = []
    nlls_dpkd = []
    nlls_dp_syn_data = []
    student_dataset = student_tokenizer("\n\n".join(dataset['test']['text']), return_tensors="pt")
    teacher_dataset = teacher_tokenizer("\n\n".join(dataset['test']['text']), return_tensors="pt")
    # we set the max length equal to the sequence length that the models were trained on
    max_length = args.sequence_len 
    stride = 128 
    seq_len = student_dataset.input_ids.size(1)
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc  - prev_end_loc
        student_input_ids = student_dataset.input_ids[:, begin_loc:end_loc].to(args.device)
        student_target_ids = student_input_ids.clone()
        student_target_ids[:, :-trg_len] = -100
        teacher_input_ids = teacher_dataset.input_ids[:, begin_loc:end_loc].to(args.device)
        teacher_target_ids = teacher_input_ids.clone()
        teacher_target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            output_pre = student_pre_trained_model(student_input_ids, labels=student_target_ids)
            output_dpsgd = student_model_dpsgd(student_input_ids, labels=student_target_ids)
            output_dpkd = student_model_dpkd(teacher_input_ids, labels=teacher_target_ids)
            output_dp_syn_data = student_model_syn(teacher_input_ids, labels=teacher_target_ids)

        nlls_pre_trained.append(output_pre.loss)
        nlls_dpsgd.append(output_dpsgd.loss)
        nlls_dpkd.append(output_dpkd.loss)
        nlls_dp_syn_data.append(output_dp_syn_data.loss)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl_pre_trained = torch.exp(torch.stack(nlls_pre_trained).mean())
    ppl_dpsgd = torch.exp(torch.stack(nlls_dpsgd).mean())
    ppl_dpkd = torch.exp(torch.stack(nlls_dpkd).mean())
    ppl_dp_syn_data = torch.exp(torch.stack(nlls_dp_syn_data).mean())

    print(f"Test set perplexity of Student model with DPKD ε = {args.target_epsilon} \
           {ppl_dpkd:.2f}")
    print(f"Test set perplexity of Student model trained with synthetic data \
          {ppl_dp_syn_data:.2f}")
    print(f"Test set perplexity of Student model trained with just DP-SGD ε = {args.target_epsilon} \
          {ppl_dpsgd:.2f}")
    print(f"Test set perplexity of pre-trained Student model \
          {ppl_pre_trained:.2f}")


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
        "--dpkd_teacher_file",
        type=str,
        default=None,
        required=True
    )
    parser.add_argument(
        "--dpkd_student_file",
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
        "--dpsgd_student_file",
        type=str,
        default=None,
        required=True
    )

    parser.add_argument(
        "--use_cache",
        type=bool,
        default=False,
        required=False
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        required=False
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:7",
        required=False
    )
    parser.add_argument(
        "--sequence_len",
        type=int,
        default=None,
        required=True
    )
    parser.add_argument(
        "--target_epsilon",
        type=float,
        default=4.0,
        required=True
    )
    args = parser.parse_args()
    main(args)