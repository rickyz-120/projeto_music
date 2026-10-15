"""
train.py  (versão 4 — v2 em versoes_antigas/train_v2.py, v1 em versoes_antigas/train_v1.py)
Treino da CNN de classificação de gêneros musicais no FMA Small.

Pré-requisito (uma vez só):   python preprocess.py
Treinar:                      python train.py
Retomar depois de Ctrl+C:     python train.py          (retoma sozinho do último checkpoint)
Começar do zero:              python train.py --fresh  (o treino anterior é renomeado, não apagado)
Teste rápido do código:       python train.py --epochs 2 --max-batches 5 --no-test --ckpt-dir tmp_ckpt --results-dir tmp_results

Novidades da v4 (melhor resultado: 66,0% na validação, contra 64,25% da v3):
  - EMA dos pesos (--ema 0.999, padrão): validação e best.pt usam uma média móvel dos
    pesos, que oscila muito menos de uma época para outra.
  - Mixup desligado por padrão (--mixup 0): com EMA + SpecAugment + label smoothing,
    o mixup virou regularização demais.
  - Opções para experimentos: --crop (tamanho do trecho de treino), --seed, --residual.

Novidades da v3 (todas configuráveis pela linha de comando):
  - Avaliação multi-recorte (--eval-crop): a validação/teste classificam trechos de ~10s
    (o mesmo tamanho do treino) e tiram a média — ver predict_logits em model.py.
  - Scheduler cosseno com aquecimento (--scheduler cosine): o learning rate sobe
    devagar nas primeiras épocas e depois desce suavemente até quase zero.
  - Mixup (--mixup 0.2): mistura duas músicas e seus rótulos (regularização).
  - Blocos com 2 convoluções (--convs-per-block 2): mais capacidade.
  - Log das épocas em results/train_log.txt.

Herdado da v2: espectrogramas pré-processados, SpecAugment + recorte aleatório,
mixed precision, AdamW, label smoothing, early stopping, checkpoints e teste no final.
"""

import os
import sys
import math
import json
import time
import random
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import FMASpecDataset
from model import GenreCNN, predict_logits

# ==== Configurações fixas ====
DATA_DIR = "data_mels"
BATCH_SIZE = 32            # treino (trechos de ~10s). Usa bem menos de 4 GB de VRAM com AMP.
EVAL_BATCH_SIZE = 16       # validação/teste usam a música inteira (30s), então batch menor
WEIGHT_DECAY = 1e-2        # AdamW: "puxa" os pesos para valores pequenos (regularização)
LABEL_SMOOTHING = 0.1      # alvo vira 0.9 na classe certa em vez de 1.0 (menos excesso de confiança)
LR_PATIENCE = 3            # (scheduler plateau) épocas sem melhorar antes de reduzir o LR
WARMUP_EPOCHS = 3          # (scheduler cosine) épocas de "aquecimento" do LR
NUM_WORKERS = 2            # processos que preparam os batches em paralelo (moderado para não pesar a CPU)
SEED = 42
CROP_FRAMES = 431          # ~10s, igual ao recorte do treino (dataset.py)

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


def make_ema_fn(decay):
    """Atualização da EMA com "aquecimento": no passo n o decaimento usado é
    min(decay, (1+n)/(10+n)). No começo (pesos ainda aleatórios) a média segue o modelo
    de perto; depois de alguns milhares de passos passa a usar o decay completo."""
    def update(ema_params, current_params, num_averaged):
        n = int(num_averaged)
        d = min(decay, (1 + n) / (10 + n))
        for e, c in zip(ema_params, current_params):
            if e.is_floating_point():
                e.lerp_(c, 1 - d)   # e = d * e + (1 - d) * c
            else:
                e.copy_(c)          # contadores inteiros do BatchNorm: só copia
    return update


def run_epoch(model, loader, criterion, optimizer=None, scaler=None, max_batches=None,
              mixup_alpha=0.0, eval_crop=None, ema=None):
    """Roda uma época. Se optimizer for passado, treina; senão, só avalia.
    ema: se passado, atualiza a média móvel dos pesos depois de cada passo de treino."""
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
                if is_train and mixup_alpha > 0:
                    # Mixup: mistura cada música com outra do batch, na proporção lam.
                    # A resposta "certa" passa a ser lam% de um gênero e (1-lam)% do outro.
                    # Isso suaviza as fronteiras entre gêneros e reduz overfitting.
                    lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                    perm = torch.randperm(inputs.size(0), device=DEVICE)
                    inputs = lam * inputs + (1 - lam) * inputs[perm]
                    outputs = model(inputs)
                    loss = lam * criterion(outputs, labels) + (1 - lam) * criterion(outputs, labels[perm])
                elif is_train:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                else:
                    # Validação/teste: média das previsões de vários trechos de ~10s
                    outputs = predict_logits(model, inputs, crop_frames=eval_crop)
                    loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                # GradScaler: multiplica a loss antes do backward para que gradientes
                # pequenos não "virem zero" em float16, e desfaz isso antes do step.
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if ema is not None:
                    ema.update_parameters(model)

            total_loss += loss.item() * inputs.size(0)
            # (com mixup, a acurácia de treino é só aproximada: comparamos com o rótulo principal)
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
    best = df.loc[df["val_acc"].idxmax()]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(df["epoch"], df["train_loss"], label="treino")
    axes[0].plot(df["epoch"], df["val_loss"], label="validação")
    axes[0].set_title("Loss")
    axes[1].plot(df["epoch"], df["train_acc"], label="treino")
    axes[1].plot(df["epoch"], df["val_acc"], label="validação")
    axes[1].scatter([best["epoch"]], [best["val_acc"]], color="red", zorder=3,
                    label=f"melhor val: {best['val_acc']:.1%} (época {int(best['epoch'])})")
    axes[1].set_title("Acurácia")
    for ax in axes:
        ax.set_xlabel("Época")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_scheduler(optimizer, kind, epochs):
    if kind == "plateau":
        # Reduz o LR pela metade quando a val_acc empaca por LR_PATIENCE épocas
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=LR_PATIENCE, min_lr=1e-5
        )

    # Cosseno com aquecimento: fator multiplicado pelo LR inicial a cada época.
    #   épocas 1..WARMUP: sobe linearmente (evita passos grandes com pesos ainda aleatórios)
    #   depois: desce em forma de cosseno até ~0 na última época (refina no final)
    def factor(e):  # e = número de épocas já concluídas (0, 1, 2, ...)
        if e < WARMUP_EPOCHS:
            return (e + 1) / WARMUP_EPOCHS
        progress = (e - WARMUP_EPOCHS) / max(1, epochs - WARMUP_EPOCHS)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def should_stop(epochs_no_improve, epoch, args):
    """Early stopping. Com o scheduler cosseno, o LR só fica pequeno no final do treino,
    que é quando o modelo costuma "refinar" e atingir o melhor resultado. Por isso,
    nesse caso só deixamos parar depois de 75% das épocas."""
    if args.scheduler == "cosine" and epoch < 0.75 * args.epochs:
        return False
    return epochs_no_improve >= args.patience


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--scheduler", choices=["cosine", "plateau"], default="cosine")
    parser.add_argument("--patience", type=int, default=15, help="early stopping (épocas sem melhorar)")
    parser.add_argument("--mixup", type=float, default=0.0, help="alpha do mixup, ex: 0.2 (0 desliga)")
    parser.add_argument("--convs-per-block", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--eval-crop", type=int, default=0,
                        help="tamanho dos recortes na avaliação (0 = música inteira de uma vez)")
    parser.add_argument("--ema", type=float, default=0.999,
                        help="decaimento da média móvel dos pesos, ex: 0.999 (0 desliga)")
    parser.add_argument("--crop", type=int, default=431,
                        help="tamanho do trecho sorteado no treino, em frames (431 = ~10s)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--residual", action="store_true", help="conexões residuais nos blocos (estilo ResNet)")
    parser.add_argument("--fresh", action="store_true", help="ignora checkpoints e começa do zero")
    parser.add_argument("--max-batches", type=int, default=None, help="limita batches por época (teste rápido)")
    parser.add_argument("--no-test", action="store_true", help="não avalia no teste ao final")
    parser.add_argument("--ckpt-dir", default="checkpoints_v4")
    parser.add_argument("--results-dir", default="results_v4")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--data-dir", default=DATA_DIR)
    args = parser.parse_args()
    eval_crop = args.eval_crop or None

    set_seed(args.seed)
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

    # Log das linhas importantes num arquivo (além do terminal)
    log_path = os.path.join(args.results_dir, "train_log.txt")

    def log(msg):
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    model_kwargs = dict(dropout=args.dropout, conv_dropout=0.1,
                        channels=(32, 64, 128, 256, 256), convs_per_block=args.convs_per_block,
                        residual=args.residual)
    config = {k: v for k, v in vars(args).items() if k not in ("fresh", "max_batches")}
    config.update(model_kwargs=model_kwargs, batch_size=BATCH_SIZE, weight_decay=WEIGHT_DECAY,
                  label_smoothing=LABEL_SMOOTHING)
    with open(os.path.join(args.results_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    log(f"Usando dispositivo: {DEVICE}"
        + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))
    log(f"Config: scheduler={args.scheduler} lr={args.lr} mixup={args.mixup} "
        f"convs_por_bloco={args.convs_per_block} dropout={args.dropout} "
        f"recorte_avaliacao={eval_crop} paciencia={args.patience} epocas_max={args.epochs} "
        f"ema={args.ema} recorte_treino={args.crop} seed={args.seed} residual={args.residual}")

    train_ds = FMASpecDataset(args.data_dir, split="training", train=True, crop_frames=args.crop)
    val_ds = FMASpecDataset(args.data_dir, split="validation", train=False)
    log(f"Treino: {len(train_ds)} faixas | Validação: {len(val_ds)} faixas")
    print(f"Gêneros ({train_ds.num_classes}): {train_ds.genres}")

    train_loader = make_loader(train_ds, BATCH_SIZE, shuffle=True, num_workers=args.workers)
    val_loader = make_loader(val_ds, EVAL_BATCH_SIZE, shuffle=False, num_workers=args.workers)

    model = GenreCNN(num_classes=train_ds.num_classes, **model_kwargs).to(DEVICE)
    log(f"Parâmetros treináveis: {sum(p.numel() for p in model.parameters()):,}")

    # EMA (média móvel exponencial dos pesos): uma "cópia calma" do modelo.
    # A cada passo: peso_ema = decay * peso_ema + (1 - decay) * peso_atual.
    # Com decay 0.999 ela lembra ~1000 passos (~5 épocas), então os "solavancos" de
    # cada passo se cancelam. A validação e o best.pt usam essa cópia.
    # use_buffers=True: também faz a média das estatísticas do BatchNorm.
    ema = None
    if args.ema > 0:
        ema = AveragedModel(model, multi_avg_fn=make_ema_fn(args.ema), use_buffers=True)
    eval_model = ema.module if ema is not None else model

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = make_scheduler(optimizer, args.scheduler, args.epochs)
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
        if ema is not None and "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt["best_val_acc"]
        epochs_no_improve = ckpt["epochs_no_improve"]
        history = ckpt["history"]
        # Recalcula a parada com as regras/paciência ATUAIS (permite estender um treino)
        early_stopped = should_stop(epochs_no_improve, ckpt["epoch"], args)
        log(f"Retomando do checkpoint: época {ckpt['epoch']} concluída, "
            f"melhor val_acc até agora = {best_val_acc:.4f}")

    if early_stopped or start_epoch > args.epochs:
        print("Este treino já terminou. Use --fresh para treinar de novo do zero "
              "(ou --epochs com um número maior para estender, se não houve early stopping).")
        return

    history_csv = os.path.join(args.results_dir, "history.csv")
    curves_png = os.path.join(args.results_dir, "curvas_treino.png")

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            t0 = time.time()
            lr = optimizer.param_groups[0]["lr"]
            train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, scaler,
                                              max_batches=args.max_batches, mixup_alpha=args.mixup,
                                              ema=ema)
            val_loss, val_acc = run_epoch(eval_model, val_loader, criterion,
                                          max_batches=args.max_batches, eval_crop=eval_crop)
            if args.scheduler == "plateau":
                scheduler.step(val_acc)
            else:
                scheduler.step()
            elapsed = time.time() - t0

            history.append(dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc,
                                val_loss=val_loss, val_acc=val_acc, lr=lr, seconds=elapsed))

            improved = val_acc > best_val_acc
            if improved:
                best_val_acc = val_acc
                epochs_no_improve = 0
                save_checkpoint({"model": eval_model.state_dict(), "model_kwargs": model_kwargs,
                                 "eval_crop": eval_crop, "epoch": epoch, "val_acc": val_acc,
                                 "genres": train_ds.genres}, best_path)
            else:
                epochs_no_improve += 1

            vram = (f" | VRAM pico {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB"
                    if DEVICE.type == "cuda" else "")
            log(f"Época {epoch:02d}/{args.epochs} | "
                f"treino: loss={train_loss:.4f} acc={train_acc:.4f} | "
                f"val: loss={val_loss:.4f} acc={val_acc:.4f} | lr={lr:.1e} | {elapsed:.0f}s{vram}"
                + ("  <- melhor modelo salvo" if improved else ""))

            stop = should_stop(epochs_no_improve, epoch, args)

            # Checkpoint completo a cada época (modelo + otimizador + scheduler + histórico)
            save_checkpoint({
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                "ema": ema.state_dict() if ema is not None else None,
                "epoch": epoch, "best_val_acc": best_val_acc,
                "epochs_no_improve": epochs_no_improve, "history": history,
                "early_stopped": stop,
            }, last_path)
            pd.DataFrame(history).to_csv(history_csv, index=False, float_format="%.5f")
            plot_history(history, curves_png)

            if stop:
                log(f"Early stopping: {args.patience} épocas sem melhorar a validação.")
                break

    except KeyboardInterrupt:
        print(f"\nInterrompido pelo usuário. Progresso salvo até a época {len(history) and history[-1]['epoch']}.")
        print("Para continuar, rode novamente o mesmo comando.")
        sys.exit(0)

    log(f"Treino concluído. Melhor acurácia de validação: {best_val_acc:.4f}")
    print(f"Histórico: {history_csv} | Curvas: {curves_png}")

    if not args.no_test and os.path.exists(best_path):
        log("\n==== Avaliação final no conjunto de TESTE (uma única vez) ====")
        import evaluate
        metrics = evaluate.main(checkpoint_path=best_path, data_dir=args.data_dir,
                                results_dir=args.results_dir)
        log(f"Acurácia no teste: {metrics['test_accuracy']:.4f}")


if __name__ == "__main__":
    main()
