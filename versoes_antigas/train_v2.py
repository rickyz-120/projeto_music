"""
train.py  (versão 2 — a versão antiga está em train_v1.py)
Treino da CNN de classificação de gêneros musicais no FMA Small.

Pré-requisito (uma vez só):   python preprocess.py
Treinar:                      python train.py
Retomar depois de Ctrl+C:     python train.py          (retoma sozinho do último checkpoint)
Começar do zero:              python train.py --fresh  (o treino anterior é renomeado, não apagado)
Teste rápido do código:       python train.py --epochs 2 --max-batches 5 --no-test --ckpt-dir tmp_ckpt --results-dir tmp_results

O que há de novo:
  - Lê espectrogramas pré-processados (muito mais rápido que decodificar mp3).
  - Data augmentation (recorte aleatório + SpecAugment) — ver dataset.py.
  - Mixed precision (float16) -> menos VRAM e mais velocidade na RTX.
  - AdamW + weight decay, label smoothing e ReduceLROnPlateau.
  - Early stopping: para sozinho se a validação não melhorar por várias épocas.
  - Checkpoint a cada época (pode interromper com Ctrl+C e retomar).
  - Salva histórico (CSV) e curvas (PNG); no fim avalia UMA vez no teste.
"""

import os
import sys
import time
import random
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# este arquivo fica em versoes_antigas/, mas usa dataset.py e model.py da pasta principal
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import FMASpecDataset
from model import GenreCNN

# ==== Configurações ====
DATA_DIR = "data_mels"
BATCH_SIZE = 32            # treino (trechos de ~10s). Usa bem menos de 4 GB de VRAM com AMP.
EVAL_BATCH_SIZE = 16       # validação/teste usam a música inteira (30s), então batch menor
EPOCHS = 40                # máximo; o early stopping costuma parar antes
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2        # AdamW: "puxa" os pesos para valores pequenos (regularização)
LABEL_SMOOTHING = 0.1      # alvo vira 0.9 na classe certa em vez de 1.0 (menos excesso de confiança)
EARLY_STOP_PATIENCE = 8    # épocas sem melhorar a val_acc antes de parar
LR_PATIENCE = 3            # épocas sem melhorar antes de reduzir o learning rate pela metade
NUM_WORKERS = 2            # processos que preparam os batches em paralelo (moderado para não pesar a CPU)
SEED = 42
MODEL_KWARGS = dict(dropout=0.4, conv_dropout=0.1, channels=(32, 64, 128, 256, 256))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    # Fixar a semente torna os resultados mais reproduzíveis entre execuções
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def worker_init_fn(worker_id):
    # Cada worker do DataLoader precisa de uma semente diferente para o numpy,
    # senão todos sorteariam os MESMOS recortes/máscaras de augmentation.
    np.random.seed((torch.initial_seed() + worker_id) % 2**32)


def make_loader(ds, batch_size, shuffle, num_workers):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=DEVICE.type == "cuda",          # acelera a cópia CPU -> GPU
        persistent_workers=num_workers > 0,        # não recria os workers a cada época (lento no Windows)
        worker_init_fn=worker_init_fn,
        drop_last=shuffle,                         # descarta o último batch incompleto só no treino
    )


def run_epoch(model, loader, criterion, optimizer=None, scaler=None, max_batches=None):
    """Roda uma época. Se optimizer for passado, treina; senão, só avalia."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    use_amp = DEVICE.type == "cuda"

    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for b, (inputs, labels) in enumerate(tqdm(loader, leave=False, desc="treino" if is_train else "val")):
            if max_batches is not None and b >= max_batches:
                break
            inputs = inputs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            # autocast: faz as contas em float16 onde é seguro (metade da memória, mais rápido)
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                # GradScaler: multiplica a loss antes do backward para que gradientes
                # pequenos não "virem zero" em float16, e desfaz isso antes do step.
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def save_checkpoint(state, path):
    # Salva primeiro num arquivo temporário e depois renomeia: se o Ctrl+C cair
    # bem no meio do salvamento, o checkpoint anterior continua íntegro.
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def plot_history(history, path):
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(df["epoch"], df["train_loss"], label="treino")
    axes[0].plot(df["epoch"], df["val_loss"], label="validação")
    axes[0].set_title("Loss")
    axes[1].plot(df["epoch"], df["train_acc"], label="treino")
    axes[1].plot(df["epoch"], df["val_acc"], label="validação")
    axes[1].set_title("Acurácia")
    for ax in axes:
        ax.set_xlabel("Época")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--fresh", action="store_true", help="ignora checkpoints e começa do zero")
    parser.add_argument("--max-batches", type=int, default=None, help="limita batches por época (teste rápido)")
    parser.add_argument("--no-test", action="store_true", help="não avalia no teste ao final")
    parser.add_argument("--ckpt-dir", default="checkpoints")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--data-dir", default=DATA_DIR)
    args = parser.parse_args()

    set_seed(SEED)
    torch.backends.cudnn.benchmark = True  # escolhe os algoritmos de convolução mais rápidos

    last_path = os.path.join(args.ckpt_dir, "last.pt")
    best_path = os.path.join(args.ckpt_dir, "best.pt")

    # --fresh: guarda o treino anterior com outro nome em vez de apagar
    if args.fresh and os.path.exists(args.ckpt_dir):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for d in (args.ckpt_dir, args.results_dir):
            if os.path.exists(d):
                os.rename(d, f"{d}_old_{stamp}")
                print(f"Treino anterior movido para {d}_old_{stamp}/")
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    print(f"Usando dispositivo: {DEVICE}"
          + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))

    train_ds = FMASpecDataset(args.data_dir, split="training", train=True)
    val_ds = FMASpecDataset(args.data_dir, split="validation", train=False)
    print(f"Treino: {len(train_ds)} faixas | Validação: {len(val_ds)} faixas")
    print(f"Gêneros ({train_ds.num_classes}): {train_ds.genres}")

    train_loader = make_loader(train_ds, BATCH_SIZE, shuffle=True, num_workers=args.workers)
    val_loader = make_loader(val_ds, EVAL_BATCH_SIZE, shuffle=False, num_workers=args.workers)

    model = GenreCNN(num_classes=train_ds.num_classes, **MODEL_KWARGS).to(DEVICE)
    print(f"Parâmetros treináveis: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # Reduz o LR pela metade quando a val_acc empaca por LR_PATIENCE épocas:
    # passos menores ajudam o modelo a "refinar" quando já está perto de um bom ponto.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=LR_PATIENCE, min_lr=1e-5
    )
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

    # ==== Retomar de onde parou ====
    start_epoch, best_val_acc, epochs_no_improve, history = 1, 0.0, 0, []
    early_stopped = False
    if os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt["best_val_acc"]
        epochs_no_improve = ckpt["epochs_no_improve"]
        history = ckpt["history"]
        early_stopped = ckpt.get("early_stopped", False)
        print(f"Retomando do checkpoint: época {ckpt['epoch']} concluída, "
              f"melhor val_acc até agora = {best_val_acc:.4f}")

    if early_stopped or start_epoch > args.epochs:
        print("Este treino já terminou. Use --fresh para treinar de novo do zero "
              "(ou --epochs com um número maior para estender, se não houve early stopping).")
        return

    history_csv = os.path.join(args.results_dir, "history.csv")
    curves_png = os.path.join(args.results_dir, "curvas_treino.png")
    epoch = start_epoch - 1

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            t0 = time.time()
            train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, scaler,
                                              max_batches=args.max_batches)
            val_loss, val_acc = run_epoch(model, val_loader, criterion, max_batches=args.max_batches)
            lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_acc)
            elapsed = time.time() - t0

            history.append(dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc,
                                val_loss=val_loss, val_acc=val_acc, lr=lr, seconds=elapsed))

            improved = val_acc > best_val_acc
            if improved:
                best_val_acc = val_acc
                epochs_no_improve = 0
                save_checkpoint({"model": model.state_dict(), "model_kwargs": MODEL_KWARGS,
                                 "epoch": epoch, "val_acc": val_acc, "genres": train_ds.genres},
                                best_path)
            else:
                epochs_no_improve += 1

            vram = (f" | VRAM pico {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB"
                    if DEVICE.type == "cuda" else "")
            print(f"Época {epoch:02d}/{args.epochs} | "
                  f"treino: loss={train_loss:.4f} acc={train_acc:.4f} | "
                  f"val: loss={val_loss:.4f} acc={val_acc:.4f} | lr={lr:.1e} | {elapsed:.0f}s{vram}"
                  + ("  <- melhor modelo salvo" if improved else ""))

            stop = epochs_no_improve >= EARLY_STOP_PATIENCE

            # Checkpoint completo a cada época (modelo + otimizador + scheduler + histórico)
            save_checkpoint({
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                "epoch": epoch, "best_val_acc": best_val_acc,
                "epochs_no_improve": epochs_no_improve, "history": history,
                "early_stopped": stop,
            }, last_path)
            pd.DataFrame(history).to_csv(history_csv, index=False, float_format="%.5f")
            plot_history(history, curves_png)

            if stop:
                print(f"Early stopping: {EARLY_STOP_PATIENCE} épocas sem melhorar a validação.")
                break

    except KeyboardInterrupt:
        print(f"\nInterrompido pelo usuário. Progresso salvo até a época {len(history) and history[-1]['epoch']}.")
        print("Para continuar, rode novamente: python train.py")
        sys.exit(0)

    print(f"Treino concluído. Melhor acurácia de validação: {best_val_acc:.4f}")
    print(f"Histórico: {history_csv} | Curvas: {curves_png}")

    if not args.no_test and os.path.exists(best_path):
        print("\n==== Avaliação final no conjunto de TESTE (uma única vez) ====")
        import evaluate_v2 as evaluate
        evaluate.main(checkpoint_path=best_path, data_dir=args.data_dir, results_dir=args.results_dir)


if __name__ == "__main__":
    main()
