"""
train.py
Loop de treino da CNN de classificação de gêneros musicais no FMA Small.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_v1 import FMASmallDataset
from model_v1 import GenreCNN

# ==== Configurações ====
AUDIO_DIR = "fma_small"
TRACKS_CSV = "fma_metadata/tracks.csv"
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(is_train):
        for inputs, labels in tqdm(loader, leave=False):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            if is_train:
                optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main():
    print(f"Usando dispositivo: {DEVICE}")

    train_ds = FMASmallDataset(AUDIO_DIR, TRACKS_CSV, split="training")
    val_ds = FMASmallDataset(AUDIO_DIR, TRACKS_CSV, split="validation")

    print(f"Treino: {len(train_ds)} faixas | Validação: {len(val_ds)} faixas")
    print(f"Gêneros ({train_ds.num_classes}): {list(train_ds.genre_to_idx.keys())}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    model = GenreCNN(num_classes=train_ds.num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion)

        print(
            f"Época {epoch:02d}/{EPOCHS} | "
            f"treino: loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val: loss={val_loss:.4f} acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pt")
            print(f"  -> novo melhor modelo salvo (val_acc={val_acc:.4f})")

    print(f"Treino concluído. Melhor acurácia de validação: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()