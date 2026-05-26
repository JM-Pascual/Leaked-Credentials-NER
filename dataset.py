import json
import random

import torch
from torch.utils.data import Dataset

from labeler import Labeler


class CredNERDataset(Dataset):
    """
    Tokenizes and aligns labels for all samples up front (eager).
    Each __getitem__ returns tensors ready for the DataLoader.
    """

    def __init__(self, samples: list[dict], labeler: Labeler):
        self.items = [labeler.align(s["tokens"], s["labels"]) for s in samples]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        return {
            "input_ids":      torch.tensor(item["input_ids"],      dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "labels":         torch.tensor(item["labels"],         dtype=torch.long),
        }

    @classmethod
    def from_jsonl(
        cls,
        path:      str,
        labeler:   Labeler,
        val_ratio: float = 0.2,
        seed:      int   = 42,
        stratify:  bool  = True,
    ) -> tuple["CredNERDataset", "CredNERDataset"]:
        samples = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                samples.append({"tokens": d["tokens"], "labels": d["labels"]})

        train_samples, val_samples = cls._split(samples, val_ratio, seed, stratify)
        return cls(train_samples, labeler), cls(val_samples, labeler)

    @staticmethod
    def _split(samples: list[dict], val_ratio: float, seed: int,
               stratify: bool) -> tuple[list, list]:
        rng = random.Random(seed)
        if stratify:
            positives = [s for s in samples if any(l != "O" for l in s["labels"])]
            negatives = [s for s in samples if all(l == "O" for l in s["labels"])]
            rng.shuffle(positives)
            rng.shuffle(negatives)
            pos_train, pos_val = CredNERDataset._split_at(positives, val_ratio)
            neg_train, neg_val = CredNERDataset._split_at(negatives, val_ratio)
            train = pos_train + neg_train
            val   = pos_val   + neg_val
            rng.shuffle(train)
            rng.shuffle(val)
            return train, val

        data = samples.copy()
        rng.shuffle(data)
        cut = int(len(data) * (1 - val_ratio))
        return data[:cut], data[cut:]

    @staticmethod
    def _split_at(lst: list, val_ratio: float) -> tuple[list, list]:
        n = int(len(lst) * (1 - val_ratio))
        return lst[:n], lst[n:]


def collate_fn(batch: list[dict], pad_token_id: int = 0) -> dict:
    """
    Pads a batch of variable-length sequences to the length of the longest one.

    Padding values:
      input_ids      → pad_token_id  (0 for bert-base-cased)
      attention_mask → 0             (model ignores these positions)
      labels         → -100          (excluded from loss)
    """
    max_len = max(item["input_ids"].shape[0] for item in batch)

    input_ids      = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels         = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, item in enumerate(batch):
        seq_len = item["input_ids"].shape[0]
        input_ids[i, :seq_len]      = item["input_ids"]
        attention_mask[i, :seq_len] = item["attention_mask"]
        labels[i, :seq_len]         = item["labels"]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}