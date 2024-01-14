import argparse
import logging
import torch
from torch.utils.data import DataLoader, Subset
import transformers
from tqdm import tqdm
import random
import numpy as np
import os
import csv
import collections

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
    if args.use_cache:
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_type, cache_dir="/data/james/.cache")
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_type)
    model = 0 
    if args.use_cache:
        if tokenizer.pad_token_id: 
            model = transformers.GPT2LMHeadModel.from_pretrained(args.model_type, cache_dir="/data/james/.cache",
                                                                pad_token_id=tokenizer.pad_token_id)
        else:
            model = transformers.GPT2LMHeadModel.from_pretrained(args.model_type, cache_dir="/data/james/.cache",
                                                                pad_token_id=tokenizer.eos_token_id)  
    else:
        if tokenizer.pad_token_id: 
            model = transformers.GPT2LMHeadModel.from_pretrained(args.model_type, pad_token_id=tokenizer.pad_token_id)
        else:
            model = transformers.GPT2LMHeadModel.from_pretrained(args.model_type, pad_token_id=tokenizer.eos_token_id)  

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

    logger.info(args)
    
    def generate_text(prompt, seq_num, prompt_length):
        ppls_cur = []
        all_data = []
        
        for _ in tqdm(range(seq_num // args.batch_size + 1)):
            input_ids = torch.tensor(prompt, device=args.device).repeat(args.batch_size, 1)
            output_sequences = model.generate(
                input_ids=input_ids,
                max_length=args.seq_len,
                temperature=args.temperature,
                top_k=args.k,
                top_p=args.p,
                early_stopping=True,
                repetition_penalty=args.repetition_penalty,
                do_sample=args.do_sample,
                num_return_sequences=args.num_return_sequences,  # overgenerate to ensure we have enough non-empty generated sequences
                no_repeat_ngram_size=2,
            )

            ppl = calc_perplexity(output_sequences, model)
            ppls_cur.append(ppl)

            generated_sequences = tokenizer.batch_decode(output_sequences, skip_special_tokens=True,
                                                         clean_up_tokenization_spaces=True)
            for g in generated_sequences:
                labels, seq = g[:prompt_length], g[:prompt_length:]
                seq = " ".join(seq.split())
                labels = labels.strip().split("\t")
                if seq:
                    all_data.append([seq] + labels)
        if len(all_data) > seq_num:
            all_data = random.sample(all_data, seq_num)
        return all_data, ppls_cur

    ppls_cur = []
    all_sequences = []
    title = 0
    with torch.no_grad():
        prompt_counter = collections.Counter()
        with open(args.input_training_file, encoding="utf-8") as rf:
            csv_reader = csv.reader(rf)
            title = next(csv_reader)
            label_column_index = [i for i, name in enumerate(title) if "label" in name]

            for line in csv_reader:
                prompt = "\t".join([line[idx] for idx in label_column_index]) + "\n\n"
                prompt_counter[prompt] += 1
        ratio_generation_training = args.total_sequences / sum(prompt_counter.values())

        for prompt_text in tqdm(prompt_counter):
            prompt = tokenizer(prompt_text)["input_ids"]
            num_seq_to_generate = round(prompt_counter[prompt_text] * ratio_generation_training)
            if num_seq_to_generate > 0:
                sequences, ppls = generate_text(prompt, num_seq_to_generate, len(prompt_text))
                all_sequences += sequences
                ppls_cur += ppls

    logger.info(f"Current PPL: %.2f±%.2f", np.mean(ppls_cur),np.std(ppls_cur))
    logger.info(f"Total generated sequences: %d", len(all_sequences))
    random.shuffle(all_sequences)

    output_name = str(args.seq_len) + "_" + args.dataset + "_synthetic_data.csv" 
    output_path = os.path.join(args.output_dir, output_name)
    with open(output_path, 'w', encoding='utf-8') as wf:
        csv_writer = csv.writer(wf)
        csv_writer.writerow(title)
        for obj in all_sequences:
            if obj[0]:
                csv_writer.writerow(obj)
        

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
        "--input_training_file",
        default=None,
        type=str,
        required=True
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
    parser.add_argument("--use_cache", type=bool)
    args = parser.parse_args()
    main(args)