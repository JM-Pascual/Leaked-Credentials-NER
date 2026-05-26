from transformers import BertForTokenClassification

from labeler import Labeler


def build_model(model_name: str, labeler: Labeler) -> BertForTokenClassification:
    """
    Load bert-base-cased with a fresh token classification head.

    HuggingFace discards the pre-training heads (MLM + NSP) and attaches
    a new Linear(768 -> num_labels) initialised with random weights.
    The Transformer body keeps its pre-trained weights.
    """
    return BertForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(labeler.label2id),
        id2label=labeler.id2label,
        label2id=labeler.label2id,
    )