import json
import os
from typing import Any, Dict, List, Tuple, Union

import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict
from torch import Tensor
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)


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


def get_corrected_tokens(tokens: List[str]) -> List[str]:
    """
    Processes a list of tokens, correcting specific tokens according to predefined rules.

    Args:
        tokens (List[str]): A list of token strings to be corrected.

    Returns:
        List[str]: A list of corrected tokens.
    """
    corrected_tokens = []
    no_action_tokens = ["NEW_LINE", "INDENT", "DEDENT"]

    for token in tokens:
        if token in no_action_tokens:
            continue
        elif token == "▁":
            corrected_tokens.append(" ")
        elif token == "STRNEWLINE":
            corrected_tokens.append("\\n")
        else:
            corrected_tokens.append(token)
    return corrected_tokens


def build_data_to_features(
    root_path: str, languages: List[str], split: str
) -> Dict[str, List[Any]]:
    """
    Processes JSONL data files for multiple programming languages and extracts relevant features.

    Args:
        root_path (str): The root directory path containing the data files.
        languages (List[str]): A list of programming languages to process.
        split (str): The data split to use (e.g., 'train', 'valid', 'test').

    Returns:
        Dict[str, List[Any]]: A dictionary containing lists of extracted features, including:
            - 'data_idx': List of data indices.
            - 'anchor_code': List of corrected docstring tokens.
            - 'positive_code': List of corrected code tokens.
            - 'src_id': List of source IDs.
            - 'src_lang': List of source languages.
            - 'trgt_id': List of target IDs.
            - 'trgt_lang': List of target languages.
            - 'language': List of languages corresponding to each data entry.
    """

    base_path = os.path.join(
        root_path, "retrieval", "code2code_search", "program_level"
    )

    idxs = []
    anchor_codes = []
    positive_codes = []
    src_ids = []
    src_langs = []
    trgt_ids = []
    trgt_langs = []
    languages_list = []

    for lang in languages:
        data_path = os.path.join(base_path, lang, f"{split}.jsonl")
        data_list = load_jsonl(data_path)

        for data in data_list:
            src, trgt = data["idx"].split("/")
            src_id, src_lang = src.split("-")
            trgt_id, trgt_lang = trgt.split("-")

            # Append data to the respective lists

            idxs.append(data["idx"])
            anchor_codes.append(
                " ".join(get_corrected_tokens(data["docstring_tokens"]))
            )
            positive_codes.append(" ".join(get_corrected_tokens(data["code_tokens"])))
            src_ids.append(src_id)
            src_langs.append(src_lang)
            trgt_ids.append(trgt_id)
            trgt_langs.append(trgt_lang)
            languages_list.append(lang)
    return {
        "data_idx": idxs,
        "query_code": anchor_codes,
        "relevant_code": positive_codes,
        "src_id": src_ids,
        "src_lang": src_langs,
        "trgt_id": trgt_ids,
        "trgt_lang": trgt_langs,
        "language": languages_list,
    }


def get_dataset(root_path: str, languages: List[str]) -> DatasetDict:
    """
    Creates a combined dataset for multiple programming languages and splits.

    Args:
        root_path (str): The root directory path containing the data files.
        languages (List[str]): A list of programming languages to process.

    Returns:
        DatasetDict: A dictionary-like object containing datasets for each data split ('train', 'val', 'test').
    """
    splits = ["train", "val", "test"]
    combined_dataset = DatasetDict(
        {
            split: Dataset.from_dict(
                build_data_to_features(root_path, languages, split)
            )
            for split in splits
        }
    )
    return combined_dataset

def collate_fn(
    batch: List[Dict[str, Any]], tokenizer: PreTrainedTokenizer
) -> Dict[str, Any]:
    """
    Collates a batch of data for training a model with anchor and positive code sequences.
    Args:
        batch (List[Dict[str, Any]]): A batch of data where each item is a dictionary containing
            "anchor_code" (str) and "positive_code" (str).
        tokenizer (PreTrainedTokenizer): A tokenizer instance from the transformers library.
    Returns:
        Dict[str, Any]: A dictionary with tokenized "query" and "relevant" code sequences, and "labels".
            - "query" (Dict[str, torch.Tensor]): Tokenized anchor codes.
            - "relevant" (Dict[str, torch.Tensor]): Tokenized positive codes.
            - "labels" (torch.Tensor): Tensor of labels corresponding to the indices of the batch items.
    """
    # remove another codes with same id even from another language
    # this is to make sure we are don't push semnatically close
    # but from diferent language and not from 'targed_language'
    unique_src_ids = set()
    unique_items = []
    for item in batch:
        if item["src_id"] not in unique_src_ids:
            unique_src_ids.add(item["src_id"])
            unique_items.append(item)
    query_codes = [item["query_code"] for item in unique_items]
    relevant_codes = [item["relevant_code"] for item in unique_items]
    
    # do tokenization
    query_codes_tensor = tokenizer(
        query_codes,
        truncation=True,
        max_length=512,
        padding="longest",
        return_tensors="pt",
    )
    relevant_codes_tensor = tokenizer(
        relevant_codes,
        truncation=True,
        max_length=512,
        padding="longest",
        return_tensors="pt",
    )
    collated_batch = {
        "query": query_codes_tensor,
        "relevant": relevant_codes_tensor,
        "labels": torch.tensor(range(len(query_codes)), dtype=torch.long),
    }
    return collated_batch

def collate_fn_concatenated(
    batch: List[Dict[str, Any]], tokenizer: PreTrainedTokenizer
) -> Dict[str, Any]:
    """
    Collates a batch of data for training a model with anchor and positive code sequences.

    Args:
        batch (List[Dict[str, Any]]): A batch of data where each item is a dictionary containing
            "anchor_code" (str) and "positive_code" (str).
        tokenizer (PreTrainedTokenizer): A tokenizer instance from the transformers library.

    Returns:
        Dict[str, Any]: A dictionary with tokenized "query" and "relevant" code sequences, and "labels".
            - "query" (Dict[str, torch.Tensor]): Tokenized anchor codes.
            - "relevant" (Dict[str, torch.Tensor]): Tokenized positive codes.
            - "labels" (torch.Tensor): Tensor of labels corresponding to the indices of the batch items.
    """

    # remove another codes with same id even from another language
    # this is to make sure we are don't push semnatically close
    # but from diferent language and not from 'targed_language'

    unique_src_ids = set()
    unique_items = []

    for item in batch:
        if item["src_id"] not in unique_src_ids:
            unique_src_ids.add(item["src_id"])
            unique_items.append(item)
    query_codes = [item["query_code"] for item in unique_items]
    relevant_codes = [item["relevant_code"] for item in unique_items]
    
    # Concatenate 'query_code' and 'relevant_code' for each item with a separator
    concatenated_sequences = [
        f"{item['query_code']} {tokenizer.sep_token} {item['relevant_code']}"
        for item in unique_items
    ]

    # do tokenization

    # Tokenize the concatenated sequences
    concatenated_tensor = tokenizer(
        concatenated_sequences,
        truncation=True,
        max_length=512,
        padding="longest",
        return_tensors="pt",
    ).cuda()

    collated_batch = {
        "input_ids": concatenated_tensor["input_ids"],
        "attention_mask": concatenated_tensor["attention_mask"],
    }

    return collated_batch


def get_pooled_embeds(
    model: PreTrainedModel, batch: Dict[str, Dict[str, torch.Tensor]], field: str
) -> torch.Tensor:
    """
    Computes the pooled embeddings for a given field in the batch using the specified model.

    Args:
        model (PreTrainedModel): A pre-trained transformer model.
        batch (Dict[str, Dict[str, Tensor]]): A batch of tokenized input data containing input IDs and attention masks.
        field (str): The key in the batch dictionary to extract the input IDs and attention masks.

    Returns:
        Tensor: The pooled embeddings of the input sequences.
    """
    ids = batch[field]["input_ids"]
    mask = batch[field]["attention_mask"]
    embeds = model(ids, attention_mask=mask)[0]
    in_mask = mask.unsqueeze(-1).expand(embeds.size()).float()

    # careful here, we only want to pool embedds when it is NOT padding

    pooled_embeds = torch.sum(embeds * in_mask, 1) / torch.clamp(
        in_mask.sum(1), min=1e-6
    )
    return pooled_embeds

class ContrastiveTrainer(Trainer):

    def compute_loss(self, model, batch, num_items_in_batch, return_outputs=False):
        a = get_pooled_embeds(model, batch, field="query")
        p = get_pooled_embeds(model, batch, field="relevant")
        assert a.shape == p.shape
        scores = torch.stack(
            [F.cosine_similarity(a_i.reshape(1, a_i.shape[0]), p, eps=1e-6) for a_i in a]
        )
        assert scores.shape[0] == scores.shape[1]
        loss = F.cross_entropy(scores, batch["labels"])
        return (loss, scores) if return_outputs else loss

'''
class ContrastiveTrainer(Trainer):
    """
    Custom Trainer class for contrastive learning using a cosine similarity-based loss function.

    This trainer overrides the `compute_loss` method to implement a custom loss function
    suitable for contrastive learning tasks.

    Methods:
        compute_loss(model, batch, return_outputs=False): Computes the custom contrastive loss.
    """

    def compute_loss(
        self,
        num_items_in_batch,
        model: PreTrainedModel,
        batch: Dict[str, Dict[str, torch.Tensor]],
        return_outputs: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Computes the contrastive loss for the given batch using the model.

        Args:
            model (PreTrainedModel): The pre-trained transformer model.
            batch (Dict[str, Dict[str, Tensor]]): A batch of data containing input IDs, attention masks, and labels.
            return_outputs (bool, optional): Whether to return the model outputs along with the loss. Default is False.

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]: The loss tensor, or a tuple of the loss tensor and the scores tensor if return_outputs is True.
        """
        a = get_pooled_embeds(model, batch, field="query")
        p = get_pooled_embeds(model, batch, field="relevant")

        assert a.shape == p.shape

        scores = torch.stack(
            [
                F.cosine_similarity(a_i.reshape(1, a_i.shape[0]), p, eps=1e-6)
                for a_i in a
            ]
        )
        assert scores.shape[0] == scores.shape[1]
        loss = F.cross_entropy(scores, batch["labels"])
        return (loss, scores) if return_outputs else loss
'''
