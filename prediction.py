import torch
import transformers
import datasets
import math


def main():
    teacher_model_6_dp = transformers.GPT2LMHeadModel.from_pretrained("gpt2-large", cache_dir="/data/james/.cache")
    pre_trained_model = transformers.GPT2LMHeadModel.from_pretrained("gpt2-large", cache_dir="/data/james/.cache") 
    student_model_syn = transformers.GPT2LMHeadModel.from_pretrained("/data/james/models/distilgpt2-DPKD-syn-data", local_files_only=True)
    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2-large", cache_dir="/data/james/.cache")
    num_added_toks = tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    mean_tok_emb = teacher_model_6_dp.transformer.wte.weight.data.mean(dim=0)
    # Initialize the newly-added token embedding to the mean of all token embeddings
    for i in range(num_added_toks):
        teacher_model_6_dp.transformer.wte.weight.data[-(i + 1), :] = mean_tok_emb

    teacher_model_6_dp.resize_token_embeddings(len(tokenizer))

    sd = torch.load("/data/james/models/gpt2-large-wikitext-6.0-dp/pytorch_model.bin")
    state_dict = {}
    for key, value in sd.items():
        key = key.replace("_module.module.", "")
        state_dict[key] = value

    teacher_model_6_dp.load_state_dict(state_dict)
    teacher_model_6_dp = teacher_model_6_dp.to("cuda:0")
    student_model_syn = student_model_syn.to("cuda:0")
    pre_trained_model = pre_trained_model.to('cuda:0')

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
    trainer_teacher_6_dp = transformers.Trainer(model=teacher_model_6_dp, args=train_args)
    trainer_student_syn = transformers.Trainer(model=student_model_syn, args=train_args)
    trainer_pre = transformers.Trainer(model=pre_trained_model, args=train_args)
    print(f"Test set perplexity of pre-trained Teacher model \
          {math.exp(trainer_pre.evaluate(eval_dataset=lm_dataset)['eval_loss']):.2f}")
    print(f"Test set perplexity of Teacher model with DP-SGD epsilon = 6 \
           {math.exp(trainer_teacher_6_dp.evaluate(eval_dataset=lm_dataset)['eval_loss']):.2f}")
    print(f"Test set perplexity of Student model trained with synthetic data \
          {math.exp(trainer_student_syn.evaluate(eval_dataset=lm_dataset)['eval_loss']):.2f}")


if __name__ == "__main__":
    main()