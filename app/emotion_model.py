import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# 1. Text preprocessing helpers
# -----------------------------

def simple_tokenize(text: str) -> List[str]:
    """
    Very simple tokenizer, similar to text preprocessing style in your RNN practical:
    lowercasing + removing non-word chars. You can swap this with a better tokenizer later.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    tokens = text.split()
    return tokens


# -----------------------------
# 2. PyTorch Dataset (for training)
# -----------------------------

class TextEmotionDataset(Dataset):
    """
    Dataset for (text, emotion_label) pairs.

    Assumes you already have:
      - a list of texts
      - a list of integer labels (0..num_classes-1)
      - a vocab dict mapping token -> index
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        vocab: Dict[str, int],
        max_len: int = 50,
        pad_idx: int = 0,
        unk_idx: int = 1,
    ):
        assert len(texts) == len(labels)
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx

    def __len__(self):
        return len(self.texts)

    def encode_text(self, text: str) -> List[int]:
        tokens = simple_tokenize(text)
        ids = [self.vocab.get(tok, self.unk_idx) for tok in tokens]
        # pad / truncate
        if len(ids) < self.max_len:
            ids += [self.pad_idx] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]
        return ids

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        label = self.labels[idx]
        ids = self.encode_text(text)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


# -----------------------------
# 3. LSTM Emotion Classifier (RNN-style)
# -----------------------------

class LSTMEmotionClassifier(nn.Module):
    """
    LSTM-based text classifier:
        Embedding -> LSTM -> last hidden state -> Fully Connected -> logits

    Pattern mirrors your RNN practical: embedding + LSTM + linear layer. 
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
        hidden_dim: int = 128,
        num_layers: int = 1,
        num_classes: int = 4,  # e.g., anger, sadness, joy, neutral
        pad_idx: int = 0,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx,
        )
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        x: (batch_size, seq_len)
        """
        embeds = self.embedding(x)            # (B, T, E)
        lstm_out, (h_n, c_n) = self.lstm(embeds)
        # h_n shape: (num_layers, B, hidden_dim)
        last_hidden = h_n[-1]                # (B, hidden_dim)
        out = self.dropout(last_hidden)
        logits = self.fc(out)                # (B, num_classes)
        return logits


# -----------------------------
# 4. Inference helper (used by FastAPI)
# -----------------------------

class EmotionInferencePipeline:
    """
    Wraps:
      - vocab
      - id2label
      - trained LSTM model
      - single-text prediction method
    """

    def __init__(
        self,
        model: LSTMEmotionClassifier,
        vocab: Dict[str, int],
        id2label: Dict[int, str],
        max_len: int = 50,
        pad_idx: int = 0,
        unk_idx: int = 1,
    ):
        self.model = model.to(DEVICE)
        self.model.eval()
        self.vocab = vocab
        self.id2label = id2label
        self.max_len = max_len
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx

    @torch.no_grad()
    def predict(self, text: str) -> Tuple[str, float]:
        """
        Returns (predicted_label_str, confidence_between_0_and_1)
        """
        tokens = simple_tokenize(text)
        ids = [self.vocab.get(tok, self.unk_idx) for tok in tokens]
        if len(ids) < self.max_len:
            ids += [self.pad_idx] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]

        x = torch.tensor([ids], dtype=torch.long, device=DEVICE)  # (1, max_len)
        logits = self.model(x)                                    # (1, num_classes)
        probs = torch.softmax(logits, dim=-1)
        conf, idx = torch.max(probs, dim=-1)
        label_idx = int(idx.item())
        label_str = self.id2label[label_idx]
        return label_str, float(conf.item())


def load_emotion_pipeline(
    checkpoint_path: str,
    vocab_path: str,
    id2label: Dict[int, str],
    embed_dim: int = 100,
    hidden_dim: int = 128,
    num_layers: int = 1,
    num_classes: int = 4,
    max_len: int = 50,
    pad_idx: int = 0,
    unk_idx: int = 1,
) -> EmotionInferencePipeline:
    """
    Helper to load trained weights + vocab and return a ready-to-use pipeline.

    You can choose how you store vocab; for simplicity, assume it's a pickled dict.
    """

    import pickle

    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    vocab_size = max(vocab.values()) + 1

    model = LSTMEmotionClassifier(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        pad_idx=pad_idx,
    )
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)

    return EmotionInferencePipeline(
        model=model,
        vocab=vocab,
        id2label=id2label,
        max_len=max_len,
        pad_idx=pad_idx,
        unk_idx=unk_idx,
    )
