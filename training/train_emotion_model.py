# train_emotion_model.py

import csv
import re
import pickle
from collections import Counter
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from app.emotion_model import (
    TextEmotionDataset,
    LSTMEmotionClassifier,
    simple_tokenize,
    DEVICE,
)

# -----------------------------
# 1. Load your labeled data
# -----------------------------
# You can adapt this to however you store your emotion dataset.
# For example, assume a CSV with columns: "text", "label"
# where label is one of: anger, sadness, joy, neutral

EMOTION2ID = {
    "anger": 0,
    "sadness": 1,
    "joy": 2,
    "neutral": 3,
}


def load_csv_data(path: str) -> Tuple[List[str], List[int]]:
    texts = []
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row["text"]
            label_str = row["label"].strip().lower()
            if label_str not in EMOTION2ID:
                continue
            texts.append(text)
            labels.append(EMOTION2ID[label_str])
    return texts, labels


# -----------------------------
# 2. Build vocabulary (RNN text practical style)
# -----------------------------

def build_vocab(texts: List[str], vocab_size: int = 10000) -> dict:
    counter = Counter()
    for t in texts:
        tokens = simple_tokenize(t)
        counter.update(tokens)

    # Reserve 0 for PAD, 1 for UNK
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for i, (tok, freq) in enumerate(counter.most_common(vocab_size - 2), start=2):
        vocab[tok] = i
    return vocab


# -----------------------------
# 3. Training loop (FCNN-style)
# -----------------------------

def train_model(
    data_csv_path: str,
    model_out_path: str = "models/emotion_lstm.pt",
    vocab_out_path: str = "models/vocab.pkl",
    batch_size: int = 64,
    lr: float = 1e-3,
    num_epochs: int = 5,
    max_len: int = 50,
):
    # 1. Load data
    texts, labels = load_csv_data(data_csv_path)
    print(f"Loaded {len(texts)} examples.")

    # 2. Build vocab
    vocab = build_vocab(texts, vocab_size=10000)
    vocab_size = max(vocab.values()) + 1
    print(f"Vocab size: {vocab_size}")

    # 3. Create Dataset
    dataset = TextEmotionDataset(
        texts=texts,
        labels=labels,
        vocab=vocab,
        max_len=max_len,
        pad_idx=0,
        unk_idx=1,
    )

    # 4. Train/val split
    val_ratio = 0.1
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 5. Model, loss, optimizer (same style as FCNN)
    model = LSTMEmotionClassifier(
        vocab_size=vocab_size,
        embed_dim=100,
        hidden_dim=128,
        num_layers=1,
        num_classes=len(EMOTION2ID),
        pad_idx=0,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 6. Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_x.size(0)

        avg_loss = total_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)

                val_loss += loss.item() * batch_x.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_x.size(0)

        val_loss /= max(1, val_total)
        val_acc = val_correct / max(1, val_total)

        print(
            f"Epoch {epoch+1}/{num_epochs} "
            f"- Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f} "
            f"- Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

    # 7. Save model + vocab
    import os

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), model_out_path)
    with open(vocab_out_path, "wb") as f:
        pickle.dump(vocab, f)

    print(f"Saved model to {model_out_path}")
    print(f"Saved vocab to {vocab_out_path}")


if __name__ == "__main__":
    # Example usage – change this path to your real CSV
    train_model(data_csv_path="data/emotion_dataset.csv")
