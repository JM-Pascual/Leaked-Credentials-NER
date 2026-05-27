import torch
from transformers import BertForTokenClassification, BertTokenizerFast

# BERT accepts up to 512 subword tokens including [CLS] and [SEP], so each
# content window is 510 subwords.
_CHUNK_SUBWORDS   = 510
# Subword overlap between consecutive windows. Words near a chunk boundary
# appear in two passes; we keep the prediction from the pass where the word
# sits closest to the centre (most bilateral context).
_OVERLAP_SUBWORDS = 50


class Inferencer:
    """
    Loads a fine-tuned checkpoint and runs token-level credential detection
    on raw code text.

    The full input is tokenized once without truncation. A sliding window then
    processes the resulting subword sequence in chunks of _CHUNK_SUBWORDS,
    adding fresh [CLS]/[SEP] tokens per chunk. This means chunking happens in
    token space, not word space, so dense content like RSA private keys never
    silently exhausts the 512-subword budget.
    """

    def __init__(self, checkpoint_dir: str = "checkpoints/cred-ner"):
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model     = BertForTokenClassification.from_pretrained(checkpoint_dir).to(self.device)
        self.tokenizer = BertTokenizerFast.from_pretrained(checkpoint_dir)
        self.id2label  = self.model.config.id2label
        self.model.eval()

    def predict(self, text: str) -> list[tuple[str, str, float]]:
        """
        Returns a list of (word, label, confidence) triples for every
        whitespace-delimited token in `text`. Labels are "O", "B-CRED", or
        "I-CRED". Confidence is the softmax probability of the predicted class.
        """
        words = text.split()
        n     = len(words)

        # ── Step 1: tokenize the full input, no truncation, no special tokens ──
        # We add [CLS]/[SEP] manually per chunk so the model always sees a
        # properly formed sequence regardless of where the window falls.
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=False,
            add_special_tokens=False,
            return_tensors="pt",
        )
        all_ids      = encoding["input_ids"][0]    # (total_subwords,)
        all_word_ids = encoding.word_ids()         # list[int] — one per subword, no Nones

        total   = len(all_ids)
        cls_id  = self.tokenizer.cls_token_id
        sep_id  = self.tokenizer.sep_token_id

        word_preds: dict[int, tuple[str, float]] = {}
        word_dists: dict[int, float]             = {}

        # ── Step 2: slide a window over the subword sequence ──────────────────
        sw_start = 0
        while sw_start < total:
            sw_end    = min(sw_start + _CHUNK_SUBWORDS, total)
            chunk_len = sw_end - sw_start

            # Build [CLS] + chunk + [SEP] and move to device.
            chunk_ids = torch.cat([
                torch.tensor([cls_id]),
                all_ids[sw_start:sw_end],
                torch.tensor([sep_id]),
            ]).unsqueeze(0).to(self.device)

            attn_mask = torch.ones_like(chunk_ids)

            # word_ids for this sequence: None for [CLS], content, None for [SEP]
            chunk_word_ids = [None] + all_word_ids[sw_start:sw_end] + [None]

            with torch.no_grad():
                logits = self.model(input_ids=chunk_ids, attention_mask=attn_mask).logits

            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)[0].tolist()

            # Centre of the content window in local subword coordinates
            # (offset by 1 because index 0 is [CLS]).
            chunk_center = (chunk_len - 1) / 2.0

            seen_local: set[int] = set()
            for subword_idx, word_idx in enumerate(chunk_word_ids):
                if word_idx is None or word_idx in seen_local:
                    continue
                seen_local.add(word_idx)

                local_pos = subword_idx - 1   # strip [CLS] offset
                dist      = abs(local_pos - chunk_center)

                if word_idx not in word_dists or dist < word_dists[word_idx]:
                    label_id               = preds[subword_idx]
                    word_dists[word_idx]   = dist
                    word_preds[word_idx]   = (
                        self.id2label[label_id],
                        probs[0, subword_idx, label_id].item(),
                    )

            if sw_end >= total:
                break
            sw_start = sw_end - _OVERLAP_SUBWORDS

        return [(words[i], *word_preds[i]) for i in range(n)]
