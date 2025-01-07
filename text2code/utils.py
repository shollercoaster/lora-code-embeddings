from typing import List, Dict, Any
from torch.utils.data import Dataset
import json
from transformers import Trainer, TrainingArguments
import torch
import torch.nn.functional as F

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return tuple(self.data[idx])

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Reads a JSON Lines (jsonl) file and returns a list of dictionaries.

    Args:
        file_path (str): The path to the jsonl file.

    Returns:
        List[Dict[str, Any]]: A list where each element is a dictionary
                              representing a JSON object from the file.
    """
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def collate_fn(batch, tokenizer):
    anchor_codes = [item[0] for item in batch]
    positive_codes = [item[1] for item in batch]

    anchor_codes_tensor = tokenizer(
        anchor_codes,
        truncation=True,
        max_length=512,
        padding="longest",
        return_tensors="pt",
    )
    positive_codes_tensor = tokenizer(
        positive_codes,
        truncation=True,
        max_length=512,
        padding="longest",
        return_tensors="pt",
    )

    collated_batch = {
        "anchor": anchor_codes_tensor,
        "positive": positive_codes_tensor,
        "labels": torch.tensor(
            range(len(anchor_codes)), dtype=torch.long
        ),
    }

    return collated_batch

def _get_pooled_embeds(model, batch, field):
    ids = batch[field]["input_ids"]
    mask = batch[field]["attention_mask"]
    embeds = model(ids, attention_mask=mask)[0]
    in_mask = mask.unsqueeze(-1).expand(embeds.size()).float()
    pooled_embeds = torch.sum(embeds * in_mask, 1) / torch.clamp(
        in_mask.sum(1), min=1e-6
    )
    return pooled_embeds

class ContrastiveTrainer(Trainer):

    def compute_loss(self, model, batch, num_items_in_batch, return_outputs=False):
        a = _get_pooled_embeds(model, batch, field="anchor")
        p = _get_pooled_embeds(model, batch, field="positive")
        scores = torch.stack(
            [F.cosine_similarity(a_i.reshape(1, a_i.shape[0]), p, eps=1e-6) for a_i in a]
        )
        # print("Shapes for pooled embeds: ", a.shape, p.shape, scores.shape)
        loss = F.cross_entropy(scores * 5, batch["labels"])
        return (loss, scores) if return_outputs else loss
