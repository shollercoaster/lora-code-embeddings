import os
from transformers import Trainer, TrainingArguments

import torch
from datasets import Dataset, DatasetDict
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, RobertaTokenizer, RobertaModel
from torch.utils.data import DataLoader
from peft import LoraConfig

from utils import load_jsonl, CustomDataset, collate_fn, ContrastiveTrainer, _get_pooled_embeds

from data_utils import create_dataset

languages = ['ruby', 'go', 'php', 'python', 'java', 'javascript']
data_dir = "../data/CSN"

def get_model(model_name):
    model = RobertaModel.from_pretrained(model_name)
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=['query', 'value'],
        lora_dropout=0.1,
        task_type="FEATURE_EXTRACTION"
    )

    for param in model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(torch.float32)

    # model.gradient_checkpointing_enable()  # reduce number of stored activations
    # model.enable_input_require_grads()

    model.add_adapter(lora_config, adapter_name="text2code-java-r64")
    model.set_adapter("text2code-java-r64")
    return model, tokenizer

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def run(model, tokenizer):
    training_args = TrainingArguments(
        "contrastive_trainer",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_steps=400,
        num_train_epochs=1,
        evaluation_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        warmup_steps=4000,
        save_strategy="epoch"
    )
    trainer = ContrastiveTrainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=lambda x: collate_fn(x, tokenizer),
    )
    trainer.train()


model_name = 'microsoft/unixcoder-base'
model, tokenizer = get_model(model_name)

for language in ['java']: # languages:
    train_dataset, eval_dataset, _, _ = create_dataset(data_dir, language)
    model, tokenizer = get_model(model_name)
    print_trainable_parameters(model)
    run(model, tokenizer)
    print(f"\n\n Training completed with {language}. \n\n")
    model.push_to_hub("text2code-java-r64")
