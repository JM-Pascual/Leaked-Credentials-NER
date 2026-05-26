import torch
from transformers import BertForTokenClassification, BertTokenizerFast


class Inferencer:
    """
    Loads a fine-tuned checkpoint and runs token-level credential detection
    on raw code text.
    """

    def __init__(self, checkpoint_dir: str = "checkpoints/cred-ner"):
        self.device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model    = BertForTokenClassification.from_pretrained(checkpoint_dir).to(self.device)
        self.tokenizer = BertTokenizerFast.from_pretrained(checkpoint_dir)
        self.id2label = self.model.config.id2label
        self.model.eval()

    def predict(self, text: str) -> list[tuple[str, str]]:
        """
        Returns a list of (word, label) pairs for every whitespace-delimited
        token in `text`. Labels are "O", "B-CRED", or "I-CRED".
        """
        words    = text.split()
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        word_ids = encoding.word_ids()

        with torch.no_grad():
            logits = self.model(
                input_ids=encoding["input_ids"].to(self.device),
                attention_mask=encoding["attention_mask"].to(self.device),
            ).logits

        preds = logits.argmax(dim=-1)[0].tolist()

        word_labels: dict[int, str] = {}
        for subword_idx, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx in word_labels:
                continue
            word_labels[word_idx] = self.id2label[preds[subword_idx]]

        return [(words[i], word_labels[i]) for i in range(len(words))]
