"""
train_transfer.py  (Parte 2 — Transfer learning)
Treina um classificador pequeno em cima dos embeddings da CNN14 pré-treinada
(gerados uma vez por panns_features.py). Como a parte pesada já foi feita, treina em segundos.

A "cabeça" tem o mesmo formato do classificador da GenreCNN da Parte 1:
    Dropout -> Linear(2048 -> hidden) -> ReLU -> Dropout -> Linear(hidden -> 8)

Data augmentation: para cada música temos 4 embeddings (música inteira + 3 trechos de 10s).
No treino, a cada época sorteamos um deles; na validação/teste usamos a música inteira.

Uso:
    python train_transfer.py                 (treina e mostra a validação)
    python train_transfer.py --hidden 512    (compara configurações pela VALIDAÇÃO)
    python train_transfer.py --test          (avaliação final no TESTE: usar uma única vez)
"""

import os
import json
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

from evaluate import plot_confusion_matrix

DATA_DIR = "data_panns"
GENRES_FROM = os.path.join("data_mels", "meta.json")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_split(short):
    emb = np.load(os.path.join(DATA_DIR, f"{short}_emb.npy")).astype(np.float32)
    labels = np.load(os.path.join(DATA_DIR, f"{short}_labels.npy"))
    valid = np.load(os.path.join(DATA_DIR, f"{short}_valid.npy"))
    return torch.from_numpy(emb[valid]), torch.from_numpy(labels[valid])


class TransferHead(nn.Module):
    def __init__(self, in_dim=2048, hidden=256, num_classes=8, dropout=0.5):
        super().__init__()
        if hidden == 0:
            # Cabeça LINEAR: uma única camada (uma "regressão logística" sobre o embedding)
            self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, num_classes))
        else:
            self.net = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
                nn.Dropout(dropout), nn.Linear(hidden, num_classes),
            )

    def forward(self, x):
        return self.net(x)


def accuracy(model, x, y, views="full"):
    """x: (N, 4, 2048). views="full" usa a música inteira; "mean" tira a média das
    probabilidades dos 4 embeddings (inteira + 3 trechos)."""
    model.eval()
    with torch.no_grad():
        if views == "full":
            pred = model(x[:, 0].to(DEVICE)).argmax(1).cpu()
        else:
            probs = torch.softmax(model(x.to(DEVICE).flatten(0, 1)), 1).view(len(x), 4, -1)
            pred = probs.mean(1).argmax(1).cpu()
    return float((pred == y).float().mean()), pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=256, help="0 = cabeça linear")
    parser.add_argument("--views", choices=["full", "mean"], default="full",
                        help="avaliação: música inteira ou média dos 4 embeddings")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test", action="store_true", help="avaliação final no teste (uma única vez)")
    parser.add_argument("--out", default="results_transfer")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    xtr, ytr = load_split("train")
    xva, yva = load_split("val")
    # Padronização (média 0, desvio 1) com estatísticas só do treino
    mean = xtr.mean(dim=(0, 1))
    std = xtr.std(dim=(0, 1)) + 1e-6
    xtr, xva = (xtr - mean) / std, (xva - mean) / std

    model = TransferHead(hidden=args.hidden, dropout=args.dropout).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc, best_state, best_epoch = 0.0, None, 0
    n = len(ytr)
    for epoch in range(1, args.epochs + 1):
        model.train()
        view = torch.randint(0, 4, (n,))           # sorteia 1 dos 4 embeddings de cada música
        x_ep = xtr[torch.arange(n), view]
        for idx in torch.randperm(n).split(64):
            xb, yb = x_ep[idx].to(DEVICE), ytr[idx].to(DEVICE)
            loss = crit(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        val_acc, _ = accuracy(model, xva, yva, args.views)
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % 20 == 0:
            print(f"Época {epoch:3d} | loss {loss.item():.3f} | val {val_acc:.4f}")

    print(f"Melhor validação: {best_acc:.4f} (época {best_epoch}) | hidden={args.hidden} dropout={args.dropout} views={args.views}")
    model.load_state_dict(best_state)
    os.makedirs(args.out, exist_ok=True)
    torch.save({"model": best_state, "mean": mean, "std": std, "args": vars(args),
                "val_acc": best_acc, "epoch": best_epoch}, os.path.join(args.out, "head.pt"))

    if not args.test:
        return
    # ==== Avaliação final no TESTE (uma única vez) ====
    with open(GENRES_FROM, encoding="utf-8") as f:
        genres = json.load(f)["genres"]
    xte, yte = load_split("test")
    acc, pred = accuracy(model, (xte - mean) / std, yte, args.views)
    y_true, y_pred = yte.numpy(), pred.numpy()
    cm = confusion_matrix(y_true, y_pred, labels=range(len(genres)))
    per_class = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=range(len(genres)),
                                                         zero_division=0)
    report = classification_report(y_true, y_pred, labels=list(range(len(genres))), target_names=genres,
                                   digits=4, zero_division=0)
    print(report)
    print(f"Acurácia no teste: {acc:.4f}")
    pd.DataFrame({"genero": genres, "acuracia": per_class, "precisao": prec, "recall": rec, "f1": f1,
                  "n_musicas": sup}).to_csv(os.path.join(args.out, "test_per_class.csv"),
                                             index=False, float_format="%.4f")
    with open(os.path.join(args.out, "test_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Acurácia no teste: {acc:.4f}\n\n{report}")
    with open(os.path.join(args.out, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"split": "test", "test_accuracy": acc, "n_test": int(len(y_true)), "val_acc": best_acc,
                   "per_class_accuracy": {g: float(a) for g, a in zip(genres, per_class)}},
                  f, indent=2, ensure_ascii=False)
    plot_confusion_matrix(cm, genres, os.path.join(args.out, "confusion_matrix.png"))
    print(f"Resultados salvos em {args.out}/")


if __name__ == "__main__":
    main()
