import os
import datasets
import dp_transformers
import transformers
import sys
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    dataset: str = field(default="wikitext", metadata={
        "help": "Name of dataset"
    })

    subset: str = field(default="wikitext-103-raw-v1", metadata={
        "help": "Subset of the dataset"
    })

    model_name: str = field(default="gpt2", metadata={
        "help": "Model name in HuggingFace, e.g. 'gpt2'"
    })

    sequence_len: int = field(default=128, metadata={
        "help": "Model sequence length"
    })

    cache_dir: str = field(default="/data/james/.cache", metadata={
        "help": "Cache directory for Huggingface data"
    })

@dataclass
class Arguments:
    train: dp_transformers.TrainingArguments
    privacy: dp_transformers.PrivacyArguments
    model: ModelArguments

def main(args: Arguments):
    #os.environ["TRANSFORMER_CACHE"] = args.model.cache_dir
    transformers.set_seed(args.train.seed)
    train_args = args.train
    privacy_args = args.privacy
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = train_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {train_args.local_rank}, device: {train_args.device}, n_gpu: {train_args.n_gpu}, "
        f"distributed training: {bool(train_args.local_rank != -1)}, 16-bits training: {train_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {train_args}")
    logger.info(f"Privacy parameters {privacy_args}")

    # Load model
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model.model_name, cache_dir=args.model.cache_dir)
    model = model.to(train_args.device)

    # Load dataset
    dataset = datasets.load_dataset(args.model.dataset, args.model.subset, cache_dir=args.model.cache_dir)

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model.model_name, cache_dir=args.model.cache_dir)
    num_added_toks = tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    mean_tok_emb = model.transformer.wte.weight.data.mean(dim=0)
    model.resize_token_embeddings(len(tokenizer))

    # Initialize the newly-added token embedding to the mean of all token embeddings
    for i in range(num_added_toks):
        model.transformer.wte.weight.data[-(i + 1), :] = mean_tok_emb

    def tokenize_function(examples):
        return tokenizer(examples['text'])

    block_size = args.model.sequence_len
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

    tokenized_datset = dataset.map(tokenize_function,
                                   batched=True,
                                   num_proc=8,
                                   remove_columns='text')
    lm_dataset = tokenized_datset.map(group_texts,
                                      batched=True,
                                      num_proc=8)

    model = model.cuda()
    model.train()
    train_args.label_names = ['labels']
    train_args.output_dir = os.path.join(train_args.output_dir, f"{args.model.model_name}-{args.model.dataset}-{args.privacy.target_epsilon}-dp")
    data_collator = dp_transformers.DataCollatorForPrivateCausalLanguageModeling(tokenizer)
    trainer = dp_transformers.dp_utils.OpacusDPTrainer(
        args=train_args,
        model=model,
        train_dataset=lm_dataset['train'],
        eval_dataset=lm_dataset['validation'],
        data_collator=data_collator,
        privacy_args=privacy_args,
        tokenizer=tokenizer,
    )

    try:
        train_result = trainer.train()
    finally:
        eps_prv = trainer.get_prv_epsilon()
        eps_rdp = trainer.get_rdp_epsilon()
        trainer.log({
            "final_epsilon_prv": eps_prv,
            "final_epsilon_rdp": eps_rdp
        })
    if train_args.local_rank == 0 or train_args.local_rank == -1:
        metrics = train_result.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

if __name__ == "__main__":
    arg_parser = transformers.HfArgumentParser((dp_transformers.TrainingArguments, dp_transformers.PrivacyArguments, ModelArguments))
    train_args, privacy_args, model_args = arg_parser.parse_args_into_dataclasses()
    main(Arguments(train=train_args, privacy=privacy_args, model=model_args))