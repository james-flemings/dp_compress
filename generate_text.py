import argparse
import logging
import torch
from torch.utils.data import DataLoader, Subset
import transformers
import datasets
from tqdm import tqdm
import random
import numpy as np
import os
import csv

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def calc_perplexity(encodings, cur_model):
    max_length = cur_model.config.n_positions
    stride = 512
    nlls_cur = []

    for i in range(0, encodings.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings[:, begin_loc:end_loc].to(args.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        target_ids[target_ids==cur_model.config.pad_token_id] = -100

        with torch.no_grad():
            outputs = cur_model(input_ids, labels=target_ids)
            nlls_cur.append(outputs[0] * trg_len)

    ppl_cur = torch.exp(torch.stack(nlls_cur).sum() / end_loc)

    return ppl_cur.item()


def main(args):
    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_type, cache_dir="/data/james/.cache")
    model = 0 
    if tokenizer.pad_token_id: 
        model = transformers.GPT2LMHeadModel.from_pretrained(args.model_type, cache_dir="/data/james/.cache",
                                                             pad_token_id=tokenizer.pad_token_id)
    else:
        model = transformers.GPT2LMHeadModel.from_pretrained(args.model_type, cache_dir="/data/james/.cache",
                                                             pad_token_id=tokenizer.eos_token_id)  
    num_added_toks = tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    mean_tok_emb = model.transformer.wte.weight.data.mean(dim=0)
    # Initialize the newly-added token embedding to the mean of all token embeddings
    for i in range(num_added_toks):
        model.transformer.wte.weight.data[-(i + 1), :] = mean_tok_emb

    model.resize_token_embeddings(len(tokenizer))

    sd = torch.load(args.pytorch_checkpoint)
    state_dict = {}
    for key, value in sd.items():
        key = key.replace("_module.module.", "")
        state_dict[key] = value

    model.load_state_dict(state_dict)
    model = model.to(args.device)
    model.eval()

    # Load dataset
    dataset = datasets.load_dataset(args.dataset, args.subset, cache_dir=args.cache_dir)

    def tokenize_function(examples):
        return tokenizer(examples['text'])

    block_size = args.seq_len
    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
        
    tokenized_dataset = dataset['train'].map(tokenize_function,
                                    batched=True,
                                    num_proc=8,
                                    remove_columns=['text'])

    lm_dataset = tokenized_dataset.map(group_texts,
                                       batched=True,
                                       num_proc=8)
    logger.info(args)
    
    def generate_text(prompt):
        input_ids = torch.tensor(prompt, device=args.device)
        output_sequence = model.generate(
            input_ids=input_ids,
            max_length=(args.prompt_len + args.seq_len),
            temperature=args.temperature,
            top_k=args.k,
            top_p=args.p,
            early_stopping=True,
            repetition_penalty=args.repetition_penalty,
            do_sample=args.do_sample,
            num_return_sequences=args.num_return_sequences,  # overgenerate to ensure we have enough non-empty generated sequences
            no_repeat_ngram_size=2,
        )

        ppl = calc_perplexity(output_sequence, model)
        return output_sequence, ppl

    lm_dataset.set_format(type='torch')
    subset = np.arange(args.total_sequences)
    data_loader = DataLoader(lm_dataset.select(subset))
    ppls_cur = []
    all_sequences = []
    all_prompts = []
    with torch.no_grad():
        for i, data in tqdm(enumerate(data_loader)):
            if i == args.total_sequences:
                break
            input = data['input_ids'].to(args.device)
            prompt = input[: , :args.prompt_len]
            sequence, ppl = generate_text(prompt) 
            all_prompts.append(tokenizer.batch_decode(prompt, skip_special_tokens=False)[0])
            all_sequences.append(tokenizer.batch_decode(sequence[:, args.prompt_len:], skip_speical_tokens=False)[0])
            print("prompt", tokenizer.batch_decode(prompt, skip_special_tokens=False)[0])
            print("generated sequence", tokenizer.batch_decode(sequence[:, args.prompt_len:], skip_speical_tokens=False)[0])
            ppls_cur.append(ppl)
    
    logger.info(f"Current PPL: %.2f±%.2f", np.mean(ppls_cur),np.std(ppls_cur))
    logger.info(f"Total generated sequences: %d", len(all_sequences))
    random.shuffle(all_sequences)

    output_path = os.path.join(args.output_dir, "synthetic_data.csv")
    fields = ['prompt', 'text', 'ppl']
    with open(output_path, 'w', encoding='utf-8') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(fields)
        for prompt, sequence, ppl in zip(all_prompts, all_sequences, ppls_cur):
            csv_writer.writerow([prompt, sequence, ppl])
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        required=True,
        help="Model to use for Synthetic text generation"
    )
    parser.add_argument(
        "--pytorch_checkpoint",
        type=str,
        default=None,
        required=True,
        help="Path to weights of trained model"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=True,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        required=True,
        help="Dataset to use for prompting"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        required=False,
        help="Data subset for Huggingface"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=128
    )
    parser.add_argument(
    "--temperature",
    type=float,
    default=1.0,
    help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--p", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", default=0)
    parser.add_argument("--num_return_sequences", type=int, default=2)
    parser.add_argument("--total_sequences", type=int, default=100000)
    parser.add_argument("--prompt_len", type=int, default=32)
    parser.add_argument("--do_sample", action="store_true", help="sampling when generation")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    main(args)