from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModel, RobertaModel, RobertaTokenizer
import sys
from utils import get_dataset, collate_fn, ContrastiveTrainer
from transformers import TrainingArguments

def get_model(model_name):
    model = RobertaModel.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = RobertaTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # named_modules/layers inside model
    # for name, module in model.named_modules():
        # print(name)
    print("\n\nNamed Parameters\n")
    for name, param in model.named_parameters():
        print(name, param.data)

    # print(model.config)
    model = model.eval().cuda()
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=['query', 'value'],
        lora_dropout=0.1
    )
    
    # model = get_peft_model(model, lora_config)
    model.add_adapter(lora_config, adapter_name="code2code-r64")
    model.set_adapter("code2code-r64")

    # model.print_trainable_parameters()
    return model, tokenizer


languages = ["C", "PHP", "Java", "C++", "C#", "Javascript", "Python"]
root_path = "../XLCoST_data"

dataset = get_dataset(root_path=root_path, languages=languages)

def run(model, tokenizer):
    training_args = TrainingArguments(
        "contrastive_trainer",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        logging_steps=200,
        num_train_epochs=1,
        evaluation_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        warmup_steps=1000,
        save_strategy="epoch"
    )
    trainer = ContrastiveTrainer(
        model,
        training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        data_collator=lambda x: collate_fn(x, tokenizer),
    )
    trainer.train()

# print("active adapter before training: ", model.active_adapters())

for model_name in ['microsoft/codebert-base', 'microsoft/graphcodebert-base', 'microsoft/unixcoder-base']:
    model, tokenizer = get_model(model_name)
    run(model, tokenizer)
    print(f"\n\n Training completed with {model_name}. \n\n")
    model.push_to_hub("code2code-r64")
