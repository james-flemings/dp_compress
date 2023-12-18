import pandas
import dp_transformers
import transformers
import logging
from dataclasses import dataclass, field
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class DistilTrainer(transformers.Trainer):
    '''
    Code grabbed from: https://huggingface.co/docs/transformers/main/tasks/knowledge_distillation_for_image_classification
    '''
    def __init__(self, teacher_model=None,
                 student_model=None,
                 temperature=None, 
                 lambda_param=None, 
                 *args, **kwargs):
        super().__init__(model=student_model, *args, **kwargs)
        self.teacher = teacher_model
        self.student = student_model 
        self.loss_function = nn.KLDivLoss(reduction='batchmean') 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.teacher.to(device)
        self.student.to(device)
        self.temperature = temperature
        self.lambda_param = lambda_param


def main(args: Arguments):
    transformers.set_seed(args.train_seed)

if __name__ == "__main__":
    arg_parser = transformers.HfArgumentParser((dp_transformers.TrainingArguments, dp_transformers.PrivacyArguments, ModelArguments))
    train_args, privacy_args, model_args = arg_parser.parse_args_into_dataclasses()
    main(Arguments(train=train_args, privacy=privacy_args, model=model_args))