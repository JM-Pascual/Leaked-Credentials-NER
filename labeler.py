from transformers import BertTokenizerFast


class Labeler:
    """
    Wraps a BertTokenizerFast and handles subword-level BIO label alignment.

    Usage:
        labeler = Labeler("bert-base-cased")
        aligned = labeler.align(tokens, labels)
    """

    def __init__(self, model_name: str, max_length: int = 512):
        _labels = ["O", "B-CRED", "I-CRED"]

        self.tokenizer  = BertTokenizerFast.from_pretrained(model_name)
        self.max_length = max_length
        self.label2id   = {l: i for i, l in enumerate(_labels)}
        self.id2label   = {i: l for i, l in enumerate(_labels)}

    def align(self, tokens: list[str], labels: list[str]) -> dict:
        """
        Tokenize a pre-split word list and align BIO labels onto subwords.

        Returns a dict with:
          input_ids      — token IDs fed to the model             (list[int])
          attention_mask — 1 for real tokens, 0 for padding       (list[int])
          labels         — label IDs; -100 where loss is ignored  (list[int])
        """
        encoding  = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        word_ids  = encoding.word_ids()
        label_ids = []
        prev_word = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != prev_word:
                label_ids.append(self.label2id[labels[word_idx]])
            else:
                label_ids.append(-100)
            prev_word = word_idx
        return {
            "input_ids":      encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels":         label_ids,
        }