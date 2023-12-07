import torch
import transformers
import datasets
import math


def main():
    model = transformers.GPT2LMHeadModel.from_pretrained("gpt2-large", cache_dir="/data/james/.cache")
    pre_trained_model = transformers.GPT2LMHeadModel.from_pretrained("gpt2-large", cache_dir="/data/james/.cache") 
    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2-large", cache_dir="/data/james/.cache")
    num_added_toks = tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    mean_tok_emb = model.transformer.wte.weight.data.mean(dim=0)
    # Initialize the newly-added token embedding to the mean of all token embeddings
    for i in range(num_added_toks):
        model.transformer.wte.weight.data[-(i + 1), :] = mean_tok_emb

    model.resize_token_embeddings(len(tokenizer))

    sd = torch.load("/data/james/models/gpt2-large-wikitext-6.0-dp/pytorch_model.bin")
    state_dict = {}
    for key, value in sd.items():
        key = key.replace("_module.module.", "")
        state_dict[key] = value

    model.load_state_dict(state_dict)
    #model = transformers.GPT2LMHeadModel._load_pretrained_model(
    #    model, 
    #)
    model = model.to("cuda:0")

    # Load dataset
    dataset = datasets.load_dataset("wikitext", "wikitext-103-raw-v1", cache_dir="/data/james/.cache")

    def tokenize_function(examples):
        return tokenizer(examples['text'])

    block_size = 512 
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
        
    tokenized_dataset = dataset['test'].map(tokenize_function,
                                    batched=True,
                                    num_proc=8,
                                    remove_columns=['text'])

    lm_dataset = tokenized_dataset.map(group_texts,
                                       batched=True,
                                       num_proc=8)
    
    train_args = transformers.TrainingArguments(output_dir="/data/james/models", per_device_eval_batch_size=8)
    trainer = transformers.Trainer(model=model, args=train_args)
    trainer_pre = transformers.Trainer(model=pre_trained_model, args=train_args)
    print(math.exp(trainer.evaluate(eval_dataset=lm_dataset)['eval_loss']))
    print(math.exp(trainer_pre.evaluate(eval_dataset=lm_dataset)['eval_loss']))


if __name__ == "__main__":
    main()