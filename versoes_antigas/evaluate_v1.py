"""
evaluate.py
Avalia o melhor modelo treinado no conjunto de validação.
"""

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from dataset_v1 import FMASmallDataset
from model_v1 import GenreCNN

AUDIO_DIR = "fma_small"
TRACKS_CSV = "fma_metadata/tracks.csv"
CHECKPOINT_PATH = "best_model.pt"
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    val_ds = FMASmallDataset(AUDIO_DIR, TRACKS_CSV, split="validation")
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = GenreCNN(num_classes=val_ds.num_classes).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    genre_names = list(val_ds.genre_to_idx.keys())
    print(classification_report(all_labels, all_preds, target_names=genre_names))


if __name__ == "__main__":
    main()