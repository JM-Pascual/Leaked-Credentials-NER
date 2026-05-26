import torch
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import BertForTokenClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from seqeval.metrics import classification_report

from src.labeler import Labeler


class Trainer:
    """
    Handles the training and validation loops for one run.

    Each call to train() starts from the base pre-trained weights (full
    retraining). The best checkpoint across epochs is saved to `checkpoint_dir`
    — overwritten each run, so you always get the best weights from the
    most recent training run.
    """

    def __init__(
        self,
        model:           BertForTokenClassification,
        train_loader:    DataLoader,
        val_loader:      DataLoader,
        labeler:         Labeler,
        epochs:          int   = 3,
        lr:              float = 2e-5,
        warmup_ratio:    float = 0.1,
        clip_norm:       float = 1.0,
        checkpoint_dir:  str   = "checkpoints/cred-ner",
    ):
        # ---------------------------------------------------------------------------
        # Device
        # Move the model to GPU if available. Batch tensors are moved in the loop.
        # ---------------------------------------------------------------------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = model.to(self.device)

        self.train_loader   = train_loader
        self.val_loader     = val_loader
        self.labeler        = labeler
        self.epochs         = epochs
        self.clip_norm      = clip_norm
        self.checkpoint_dir = Path(checkpoint_dir)

        # ---------------------------------------------------------------------------
        # Optimizer — AdamW
        #
        # AdamW is Adam with weight decay applied correctly: it decouples the
        # weight decay from the gradient update, which Adam's original formulation
        # conflates. This improves generalisation when fine-tuning large models.
        # lr=2e-5 is the standard starting point for BERT fine-tuning — small
        # enough to not destroy the pre-trained weights, large enough to train
        # the new classification head quickly.
        # ---------------------------------------------------------------------------
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

        # ---------------------------------------------------------------------------
        # Scheduler — linear warmup then linear decay
        #
        # For the first `warmup_steps` steps the learning rate rises linearly
        # from 0 to lr. After that it decays linearly back to 0 by the last step.
        #
        # Why warmup? The classification head starts with random weights. If the
        # full learning rate hits from step 1 the head produces wild gradients
        # that can corrupt BERT's pre-trained representations before the head
        # has had a chance to stabilise. Warmup gives the head a few steps to
        # reach a reasonable state before full-speed updates begin.
        # ---------------------------------------------------------------------------
        total_steps  = len(train_loader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    def _move(self, batch: dict) -> dict:
        return {k: v.to(self.device) for k, v in batch.items()}

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in self.train_loader:
            batch = self._move(batch)

            # Forward pass — BertForTokenClassification computes cross-entropy
            # loss internally when labels are provided.
            outputs = self.model(**batch)
            loss    = outputs.loss

            # Backward pass
            loss.backward()

            # Gradient clipping — caps the global norm of all gradients to
            # clip_norm (1.0). Prevents a single bad batch from taking a
            # destructively large step and unwinding pre-trained weights.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def _eval_epoch(self) -> tuple[float, str]:
        self.model.eval()
        total_loss  = 0.0
        all_preds   = []
        all_trues   = []

        with torch.no_grad():
            for batch in self.val_loader:
                batch   = self._move(batch)
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()

                # logits: (batch, seq_len, num_labels) → argmax → (batch, seq_len)
                predictions = outputs.logits.argmax(dim=-1)
                true_labels = batch["labels"]

                # Decode each sequence in the batch, skipping -100 positions.
                # -100 marks special tokens ([CLS], [SEP]) and non-first subwords
                # — positions we never want to evaluate.
                for pred_seq, true_seq in zip(predictions, true_labels):
                    pred_str, true_str = [], []
                    for p, t in zip(pred_seq, true_seq):
                        if t.item() == -100:
                            continue
                        pred_str.append(self.labeler.id2label[p.item()])
                        true_str.append(self.labeler.id2label[t.item()])
                    all_preds.append(pred_str)
                    all_trues.append(true_str)

        avg_loss = total_loss / len(self.val_loader)
        report   = classification_report(all_trues, all_preds, zero_division=0)
        return avg_loss, report

    def train(self) -> None:
        print(f"Training on {self.device}\n")
        best_val_loss  = float("inf")
        best_report    = ""

        for epoch in range(1, self.epochs + 1):
            train_loss        = self._train_epoch()
            val_loss, report  = self._eval_epoch()

            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                best_report   = report
                self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                self.model.save_pretrained(self.checkpoint_dir)
                self.labeler.tokenizer.save_pretrained(self.checkpoint_dir)

            marker = "  ✓ saved" if improved else ""
            print(f"Epoch {epoch}/{self.epochs}  "
                  f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}{marker}")

        print(f"\nBest val_loss: {best_val_loss:.4f} — checkpoint: {self.checkpoint_dir}")
        print(f"\nEvaluation report (best epoch):\n{best_report}")