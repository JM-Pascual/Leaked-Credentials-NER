"""
NER training pipeline for hardcoded credential detection.
Base model: bert-base-cased (encoder-only, bidirectional, case-sensitive)
"""

from torch.utils.data import DataLoader

from src.labeler import Labeler
from src.dataset import CredNERDataset, collate_fn
from src.model import build_model
from src.trainer import Trainer

MODEL_NAME = "bert-base-cased"
BATCH_SIZE = 8
EPOCHS     = 3
VAL_RATIO  = 0.2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    labeler = Labeler(MODEL_NAME)

    train_dataset, val_dataset = CredNERDataset.from_jsonl(
        "data/train/data-enriched.jsonl", labeler, VAL_RATIO
    )
    print(f"Train samples: {len(train_dataset)}  |  Val samples: {len(val_dataset)}")

    pad_id   = labeler.tokenizer.pad_token_id
    _collate = lambda b: collate_fn(b, pad_id)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=_collate)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=_collate)

    model   = build_model(MODEL_NAME, labeler)
    trainer = Trainer(model, train_loader, val_loader, labeler, epochs=EPOCHS)
    trainer.train()