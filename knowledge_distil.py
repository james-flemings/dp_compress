import dp_transformers
import transformers
import logging
from dataclasses import dataclass, field
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import datasets

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    dataset: str = field(default="yelp", metadata={
        "help": "Name of dataset"
    })

    student_model: str = field(default="distilgpt2", metadata={
        "help": "Model name in HuggingFace, e.g. 'gpt2'"
    })

    teacher_model: str = field(default="gpt2", metadata={
        "help": "Model name in HuggingFace, e.g. 'gpt2'"
    })

    sequence_len: int = field(default=128, metadata={
        "help": "Model sequence length"
    })

    use_cache: bool = field(default=False, metadata={
        "help": "Whether to use cache directory"
    })

    cache_dir: str = field(default="/data/james/.cache", metadata={
        "help": "Cache directory for Huggingface data"
    })
    
    pytorch_checkpoint: str = field(default="/data/james/models", metadata={
        "help": "Pytorch checkpoint for teacher model"
    })

    lambda_param: float = field(default=0.0, metadata={
        "help": "Weight parameter for Student Teacher loss"
    })

    alpha_cos: float = field(default=0.0, metadata={
        "help": "Weight parameter for Cosine loss"
    })

    temperature: float = field(default=1.0, metadata={
        "help": "Temperature parameter"
    })

    synthetic_data_file: str = field(default="data/james/synthetic_data.csv", metadata={
        "help": "File path to synthetic dataset"
    })

@dataclass
class Arguments:
    train: dp_transformers.TrainingArguments
    privacy: dp_transformers.PrivacyArguments
    model: ModelArguments

class DistilTrainer(transformers.Trainer):
    '''
    Code grabbed from: https://huggingface.co/docs/transformers/main/tasks/knowledge_distillation_for_image_classification
    '''
    def __init__(self, teacher_model=None,
                 student_model=None,
                 temperature=0, 
                 lambda_param=0,
                 alpha_cos=0,
                 *args, **kwargs):
        super().__init__(model=student_model, *args, **kwargs)
        self.teacher = teacher_model
        self.student = student_model 
        self.loss_function = nn.KLDivLoss(reduction='batchmean') 
        self.temperature = temperature
        self.lambda_param = lambda_param
        self.alpha_cos = alpha_cos
        #self.hidden_loss_fct = nn.CosineEmbeddingLoss(reduction="mean")
        self.hidden_loss_fct = nn.MSELoss(reduction="mean")

    def compute_loss(self, model, inputs, return_outputs=False):
        '''
        Knowledge Distillation from https://arxiv.org/pdf/1503.02531.pdf 
        '''
        student_output = self.student(**inputs, output_hidden_states=True)


        # Compute soft targets for teacher and student
        soft_student = F.log_softmax(student_output.logits / self.temperature, dim=-1)

        loss = 0
        with torch.no_grad():
            teacher_output = self.teacher(**inputs, output_hidden_states=True)
        # Compute the loss
        if self.lambda_param > 0:
            soft_teacher = F.softmax(teacher_output.logits / self.temperature, dim=-1)
            distillation_loss = self.loss_function(soft_student[:, 5:], soft_teacher[:, 5:]) * (self.temperature ** 2)
            loss += self.lambda_param * distillation_loss

        if self.alpha_cos > 0:
            s_hidden_states = student_output['hidden_states'][-1]
            t_hidden_states = teacher_output['hidden_states'][-1]
            attention_mask = inputs['attention_mask']
            mask = (attention_mask.unsqueeze(-1).expand_as(s_hidden_states) > 0)
            assert s_hidden_states.size() == t_hidden_states.size()
            dim = s_hidden_states.size(-1) 

            s_hidden_states_slct = s_hidden_states.view(-1, dim)
            t_hidden_states_slct = t_hidden_states.view(-1, dim)

            target = torch.ones(s_hidden_states_slct.size(0)).to(s_hidden_states_slct.device)
            loss_hid = self.hidden_loss_fct(s_hidden_states_slct, t_hidden_states_slct)
            loss += self.alpha_cos * loss_hid

        # Compute the true label loss
        student_target_loss = student_output.loss

        # Calculate final loss
        loss += (1-self.lambda_param) * student_target_loss  

        return (loss, student_output) if return_outputs else loss

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

    if args.model.use_cache:
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model.teacher_model, cache_dir=args.model.cache_dir)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model.teacher_model)

    teacher_model = 0 
    if tokenizer.pad_token_id: 
        if args.model.use_cache:
            teacher_model = transformers.GPT2LMHeadModel.from_pretrained(args.model.teacher_model, cache_dir=args.model.cache_dir,
                                                             pad_token_id=tokenizer.pad_token_id)
        else:
            teacher_model = transformers.GPT2LMHeadModel.from_pretrained(args.model.teacher_model,
                                                             pad_token_id=tokenizer.pad_token_id)
    else:
        if args.model.use_cache:
            teacher_model = transformers.GPT2LMHeadModel.from_pretrained(args.model.teacher_model, cache_dir=args.model.cache_dir,
                                                             pad_token_id=tokenizer.eos_token_id)  
        else:
            teacher_model = transformers.GPT2LMHeadModel.from_pretrained(args.model.teacher_model,
                                                             pad_token_id=tokenizer.eos_token_id)  
    # Load student model
    if args.model.use_cache:
        student_model = transformers.GPT2LMHeadModel.from_pretrained(args.model.student_model, cache_dir=args.model.cache_dir)
    else:
        student_model = transformers.GPT2LMHeadModel.from_pretrained(args.model.student_model)

    num_added_toks = tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    teacher_mean_tok_emb = teacher_model.transformer.wte.weight.data.mean(dim=0)
    student_mean_tok_emb = student_model.transformer.wte.weight.data.mean(dim=0)
    # Initialize the newly-added token embedding to the mean of all token embeddings
    for i in range(num_added_toks):
        teacher_model.transformer.wte.weight.data[-(i + 1), :] = teacher_mean_tok_emb
        student_model.transformer.wte.weight.data[-(i + 1), :] = student_mean_tok_emb

    teacher_model.resize_token_embeddings(len(tokenizer))
    student_model.resize_token_embeddings(len(tokenizer))

    # Load DP Weights
    sd = torch.load(args.model.pytorch_checkpoint, map_location='cpu')
    state_dict = {}
    for key, value in sd.items():
        key = key.replace("_module.module.", "")
        state_dict[key] = value

    teacher_model.load_state_dict(state_dict)
    teacher_model.cuda()
    teacher_model.eval()
    del state_dict

    student_model.cuda()
    student_model.train()

    if args.model.use_cache:
        dataset = datasets.load_dataset('csv', data_files={'train': args.model.synthetic_data_file}, cache_dir=args.model.cache_dir)
    else:
        dataset = datasets.load_dataset('csv', data_files={'train': args.model.synthetic_data_file})

    dataset = dataset['train'].train_test_split(test_size=0.01)
    label_column_names = [name for name in dataset["train"].column_names if "label" in name]

    # Tokenize data
    def preprocess_function(examples):
        batch = []
        '''
        for t in range(len(examples['text'])):
            text = "\t".join([examples[name][t] for name in label_column_names]) + "\n\n" + examples['text'][t] + tokenizer.eos_token
            batch.append(text)
        '''
        batch = examples['text']
        result = tokenizer(batch, padding="max_length", truncation=True,
                           max_length=args.model.sequence_len)

        result["labels"] = result["input_ids"].copy()
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
    #train_args.label_names = ['input_ids']
    output_name = f'{args.model.student_model}-{args.model.dataset}-{privacy_args.target_epsilon}-DPKD-syn-data'
    train_args.output_dir = os.path.join(train_args.output_dir, output_name)
    if args.model.lambda_param > 0:
        trainer = DistilTrainer(
            student_model=student_model,
            teacher_model=teacher_model,
            args=train_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset['test'],
            data_collator=transformers.DefaultDataCollator(),
            tokenizer=tokenizer,
            temperature=args.model.temperature,
            lambda_param=args.model.lambda_param,
            alpha_cos=args.model.alpha_cos
        )
    else:
        train_args.label_names = ['labels']
        trainer = transformers.Trainer(
        args=train_args,
        model=student_model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        data_collator=transformers.DefaultDataCollator(),
        tokenizer=tokenizer
    )
    train_result = trainer.train()
    if train_args.local_rank == 0 or train_args.local_rank == -1:
        metrics = train_result.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)        

if __name__ == "__main__":
    arg_parser = transformers.HfArgumentParser((dp_transformers.TrainingArguments, dp_transformers.PrivacyArguments, ModelArguments))
    train_args, privacy_args, model_args = arg_parser.parse_args_into_dataclasses()
    main(Arguments(train=train_args, privacy=privacy_args, model=model_args))