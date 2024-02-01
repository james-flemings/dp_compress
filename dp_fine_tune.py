import os
import sys
import logging
import datasets
import dp_transformers
import transformers
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    data_dir: str = field(default="./", metadata={
        "help": "Path to dataset"
    })

    dataset_name: str = field(default="yelp", metadata={
        "help": "Name of dataset used"
    })

    model_name: str = field(default="gpt2", metadata={
        "help": "Model name in HuggingFace, e.g. 'gpt2'"
    })

    sequence_len: int = field(default=128, metadata={
        "help": "Model sequence length"
    })
    use_cc: bool = field(default=False, metadata={
        "help": "Whether to use control codes"
    })
    use_cache: bool = field(default=False, metadata={
        "help": "Whether to use cache directory or not"
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
    if args.model.use_cache:
        model = transformers.AutoModelForCausalLM.from_pretrained(args.model.model_name, cache_dir=args.model.cache_dir)
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(args.model.model_name)

    # Load dataset
    data_path_train = os.path.join(args.model.data_dir, "train.csv")
    data_path_val = os.path.join(args.model.data_dir, "val.csv")
    if args.model.use_cache:
        dataset = datasets.load_dataset('csv', data_files={'train': data_path_train, 'validation': data_path_val}, cache_dir=args.model.cache_dir)
    else:
        dataset = datasets.load_dataset('csv', data_files={'train': data_path_train, 'validation': data_path_val})

    # Load tokenizer
    if args.model.use_cache:
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model.model_name, cache_dir=args.model.cache_dir)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model.model_name)
    num_added_toks = tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    mean_tok_emb = model.transformer.wte.weight.data.mean(dim=0)
    model.resize_token_embeddings(len(tokenizer))

    # Initialize the newly-added token embedding to the mean of all token embeddings
    for i in range(num_added_toks):
        model.transformer.wte.weight.data[-(i + 1), :] = mean_tok_emb

    label_column_names = [name for name in dataset["train"].column_names if "label" in name]

    # Tokenize data
    def preprocess_function(examples):
        batch = []
        if args.model.use_cc:
            for t in range(len(examples['text'])):
                text = "\t".join([examples[name][t] for name in label_column_names]) + "\n\n" + examples['text'][t] + tokenizer.eos_token
                batch.append(text)
        else:
            batch = examples['text']
        result = tokenizer(batch, padding="max_length", truncation=True,
                           max_length=args.model.sequence_len)

        return result

    # Tokenize data
    with train_args.main_process_first(desc="tokenizing dataset"):
        dataset = dataset.map(
            preprocess_function,
            batched=True, 
            num_proc=args.train.dataloader_num_workers,
            desc="tokenizing dataset",
            remove_columns=dataset.column_names['train']
        )

    model = model.cuda()
    model.train()
    train_args.label_names = ['labels']
    if args.model.use_cc:
        dir_name = f"cc-{args.model.model_name}-{args.model.dataset_name}-{args.privacy.target_epsilon}-dp"
    else:
        dir_name = f"{args.model.model_name}-{args.model.dataset_name}-{args.privacy.target_epsilon}-dp"
    train_args.output_dir = os.path.join(train_args.output_dir, dir_name)
    data_collator = dp_transformers.DataCollatorForPrivateCausalLanguageModeling(tokenizer)
    trainer = dp_transformers.dp_utils.OpacusDPTrainer(
        args=train_args,
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
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