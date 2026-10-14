"""
evaluate.py  (versão 2 — a versão antiga está em evaluate_v1.py)
Avaliação FINAL do melhor modelo no conjunto de TESTE.

IMPORTANTE: o teste deve ser usado UMA ÚNICA VEZ, no final. Se usarmos o teste para
escolher hiperparâmetros, ele deixa de ser uma medida honesta do desempenho em
músicas "nunca vistas". Para ajustes, use sempre a validação.

O train.py chama este script automaticamente ao terminar. Também pode rodar sozinho:
    python evaluate.py

Gera na pasta results/:
    test_metrics.json           -> acurácia geral + acurácia por gênero
    test_per_class.csv          -> precisão, recall, f1 e acurácia por gênero
    test_classification_report.txt
    confusion_matrix.png        -> matriz de confusão (contagens e %)
"""

import os
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib
matplotlib.use("Agg")  # gera imagens sem abrir janela
import matplotlib.pyplot as plt

# este arquivo fica em versoes_antigas/, mas usa dataset.py e model.py da pasta principal
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import FMASpecDataset
from model import GenreCNN

DATA_DIR = "data_mels"
CHECKPOINT_PATH = os.path.join("checkpoints", "best.pt")
RESULTS_DIR = "results"
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(model, loader, device):
    model.eval()
    use_amp = device.type == "cuda"
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                outputs = model(inputs)
            all_preds.append(outputs.argmax(dim=1).cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_labels), np.concatenate(all_preds)


def plot_confusion_matrix(cm, genres, path):
    # Normaliza por linha: cada linha mostra, para um gênero verdadeiro,
    # a % de músicas que o modelo previu como cada gênero.
    cm_pct = cm / cm.sum(axis=1, keepdims=True).clip(min=1) * 100

    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    fig.colorbar(im, ax=ax, label="% das músicas do gênero verdadeiro")
    ax.set_xticks(range(len(genres)), genres, rotation=45, ha="right")
    ax.set_yticks(range(len(genres)), genres)
    ax.set_xlabel("Gênero previsto")
    ax.set_ylabel("Gênero verdadeiro")
    ax.set_title("Matriz de confusão — conjunto de teste")
    for i in range(len(genres)):
        for j in range(len(genres)):
            color = "white" if cm_pct[i, j] > 50 else "black"
            ax.text(j, i, f"{cm[i, j]}\n{cm_pct[i, j]:.0f}%", ha="center", va="center",
                    fontsize=8, color=color)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(checkpoint_path=CHECKPOINT_PATH, data_dir=DATA_DIR, results_dir=RESULTS_DIR):
    os.makedirs(results_dir, exist_ok=True)

    test_ds = FMASpecDataset(data_dir, split="test", train=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=DEVICE.type == "cuda")
    genres = test_ds.genres

    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model = GenreCNN(num_classes=test_ds.num_classes, **ckpt.get("model_kwargs", {})).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    print(f"Modelo carregado de {checkpoint_path} (época {ckpt.get('epoch')}, "
          f"val_acc={ckpt.get('val_acc', float('nan')):.4f})")
    print(f"Avaliando no TESTE: {len(test_ds)} faixas")

    y_true, y_pred = predict(model, test_loader, DEVICE)

    acc = float((y_true == y_pred).mean())
    cm = confusion_matrix(y_true, y_pred, labels=range(len(genres)))
    # Acurácia por classe = acertos do gênero / total de músicas daquele gênero
    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(genres)), zero_division=0
    )

    report = classification_report(y_true, y_pred, labels=list(range(len(genres))),
                                   target_names=genres, digits=4, zero_division=0)
    print(report)
    print(f"Acurácia no teste: {acc:.4f}")

    per_class = pd.DataFrame({
        "genero": genres,
        "acuracia": per_class_acc,
        "precisao": prec,
        "recall": rec,
        "f1": f1,
        "n_musicas": support,
    })
    per_class.to_csv(os.path.join(results_dir, "test_per_class.csv"), index=False, float_format="%.4f")

    with open(os.path.join(results_dir, "test_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Acurácia no teste: {acc:.4f}\n\n{report}")

    metrics = {
        "test_accuracy": acc,
        "n_test": int(len(y_true)),
        "per_class_accuracy": {g: float(a) for g, a in zip(genres, per_class_acc)},
        "checkpoint_epoch": ckpt.get("epoch"),
        "best_val_accuracy": ckpt.get("val_acc"),
    }
    with open(os.path.join(results_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    plot_confusion_matrix(cm, genres, os.path.join(results_dir, "confusion_matrix.png"))
    print(f"Resultados salvos em {results_dir}/")
    return metrics


if __name__ == "__main__":
    main()
