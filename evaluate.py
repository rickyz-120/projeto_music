"""
evaluate.py  (versão 4 — versões antigas em versoes_antigas/)
Avaliação FINAL no conjunto de TESTE de um modelo ou de um ENSEMBLE (média de vários modelos).

IMPORTANTE: o teste deve ser usado UMA ÚNICA VEZ, no final. Se usarmos o teste para
escolher hiperparâmetros, ele deixa de ser uma medida honesta do desempenho em
músicas "nunca vistas". Para ajustes, use sempre a validação.

O train.py chama este script automaticamente ao terminar. Também pode rodar sozinho:
    python evaluate.py                                   (checkpoints_v4/best.pt)
    python evaluate.py --ckpt A.pt B.pt --out results_final  (ensemble de A e B)
    python evaluate.py --ckpt A.pt --split validation --out tmp  (comparar na validação)

Gera na pasta de resultados:
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

from dataset import FMASpecDataset
from model import GenreCNN, predict_logits

DATA_DIR = "data_mels"
CHECKPOINT_PATH = os.path.join("checkpoints_v4", "best.pt")
RESULTS_DIR = "results_v4"
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_probs(model, loader, device, eval_crop=None):
    """Retorna (rótulos verdadeiros, probabilidades de cada gênero) para todo o loader."""
    model.eval()
    use_amp = device.type == "cuda"
    all_probs, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                # mesma forma de avaliar usada na validação (multi-recorte ou 30s inteiros)
                outputs = predict_logits(model, inputs, crop_frames=eval_crop)
            all_probs.append(outputs.float().exp().cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_labels), np.concatenate(all_probs)


def load_model(path, num_classes):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model = GenreCNN(num_classes=num_classes, **ckpt.get("model_kwargs", {})).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    print(f"Modelo carregado de {path} (época {ckpt.get('epoch')}, "
          f"val_acc={ckpt.get('val_acc', float('nan')):.4f})")
    return model, ckpt


def plot_confusion_matrix(cm, genres, path, split_pt="teste"):
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
    ax.set_title(f"Matriz de confusão — conjunto de {split_pt}")
    for i in range(len(genres)):
        for j in range(len(genres)):
            color = "white" if cm_pct[i, j] > 50 else "black"
            ax.text(j, i, f"{cm[i, j]}\n{cm_pct[i, j]:.0f}%", ha="center", va="center",
                    fontsize=8, color=color)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(checkpoint_path=CHECKPOINT_PATH, data_dir=DATA_DIR, results_dir=RESULTS_DIR, split="test"):
    """checkpoint_path pode ser um caminho ou uma LISTA de caminhos.
    Com uma lista, avalia o ENSEMBLE: a média das probabilidades de todos os modelos
    (modelos diferentes erram em músicas diferentes, então a média costuma acertar mais).
    split="validation" avalia na validação: serve para COMPARAR modelos/ensembles sem gastar o teste."""
    split_pt = "teste" if split == "test" else "validação"
    os.makedirs(results_dir, exist_ok=True)
    paths = [checkpoint_path] if isinstance(checkpoint_path, str) else list(checkpoint_path)

    test_ds = FMASpecDataset(data_dir, split=split, train=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=DEVICE.type == "cuda")
    genres = test_ds.genres
    print(f"Avaliando em {split_pt.upper()}: {len(test_ds)} faixas")

    probs_list, individual = [], {}
    for p in paths:
        model, ckpt = load_model(p, test_ds.num_classes)
        y_true, probs = predict_probs(model, test_loader, DEVICE, eval_crop=ckpt.get("eval_crop"))
        probs_list.append(probs)
        individual[p] = float((probs.argmax(axis=1) == y_true).mean())
        del model
    if len(paths) > 1:
        for p, a in individual.items():
            print(f"  acurácia individual de {p}: {a:.4f}")
    y_pred = np.mean(probs_list, axis=0).argmax(axis=1)

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
    print(f"Acurácia em {split_pt}: {acc:.4f}")

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
        "split": split,
        "test_accuracy": acc,
        "n_test": int(len(y_true)),
        "per_class_accuracy": {g: float(a) for g, a in zip(genres, per_class_acc)},
        "checkpoints": paths,
        "individual_test_accuracy": individual,
    }
    with open(os.path.join(results_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    plot_confusion_matrix(cm, genres, os.path.join(results_dir, "confusion_matrix.png"), split_pt)
    print(f"Resultados salvos em {results_dir}/")
    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", nargs="+", default=[CHECKPOINT_PATH], help="um ou mais checkpoints")
    parser.add_argument("--out", default=RESULTS_DIR)
    parser.add_argument("--split", choices=["test", "validation"], default="test",
                        help="validation = comparar modelos sem gastar o teste")
    args = parser.parse_args()
    main(checkpoint_path=args.ckpt, results_dir=args.out, split=args.split)
