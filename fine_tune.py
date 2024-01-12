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

    cache_dir: str = field(default="/data/james/.cache", metadata={
        "help": "Cache directory for Huggingface data"
    })
    use_control_codes: bool = field(default=True, metadata={
        "help": "Prepend control codes to text"
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
    data_path_train = os.path.join(args.model.data_dir, "train.csv")
    data_path_val = os.path.join(args.model.data_dir, "val.csv")
    dataset = datasets.load_dataset('csv', data_files={'train': data_path_train, 'validation': data_path_val}, cache_dir=args.model.cache_dir)

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model.model_name, cache_dir=args.model.cache_dir)
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
        for t in range(len(examples['text'])):
            text = "\t".join([examples[name][t] for name in label_column_names]) + "\n\n" + examples['text'][t] + tokenizer.eos_token
            batch.append(text)

        result = tokenizer(batch, padding="max_length", truncation=True,
                           max_length=args.model.sequence_len)

        return result

    def tokenize(examples):
        return tokenizer(examples['text'])

    block_size = args.model.sequence_len
    def group_function(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Tokenize data
    with train_args.main_process_first(desc="tokenizing dataset"):
        print(args.model.use_control_codes)
        if args.model.use_control_codes:
            dataset = dataset.map(
                preprocess_function, batched=True, desc="tokenizing dataset", remove_columns=dataset.column_names['train']
            )
        else:
            dataset = dataset.map(tokenize, remove_columns=dataset.column_names['train'], desc="Tokenizing dataset")
            dataset = dataset.map(group_function, batched=True, desc="grouping dataset")

    model = model.cuda()
    model.train()
    train_args.label_names = ['labels']
    train_args.output_dir = os.path.join(train_args.output_dir, f"{args.model.model_name}-{args.model.dataset_name}-{args.privacy.target_epsilon}-dp")
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