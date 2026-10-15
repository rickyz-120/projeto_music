"""
gerar_figuras.py
Gera imagens prontas para a apresentação a partir dos resultados REAIS do treino
(nenhum número é inventado: tudo é lido de history.csv, test_metrics.json e train_log.txt).

Uso:
    python gerar_figuras.py --final results_final --train-dir results_v3a
    python gerar_figuras.py --final results_v4 --out docs/img     (figuras do README)

Saída em apresentacao/:
    01_curvas_treino.png        -> loss e acurácia por época (treino x validação)
    02_comparacao_versoes.png   -> acurácia de validação e teste de cada versão
    03_acuracia_por_genero.png  -> acurácia no teste por gênero
    04_matriz_confusao.png      -> cópia da matriz de confusão do teste
    05_log_treino.png           -> "print" do terminal, renderizado a partir do log real
"""

import os
import json
import shutil
import argparse

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "apresentacao"

# Paleta categórica validada (azul, laranja) + tons neutros para texto/grade
BLUE, ORANGE = "#2a78d6", "#eb6834"
TEXT, TEXT_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2,
    "xtick.color": TEXT_2, "ytick.color": TEXT_2, "axes.titleweight": "bold",
    "axes.titlesize": 13, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "legend.frameon": False, "figure.facecolor": "white",
})


def pct(v, casas=1):
    """0.6425 -> '64,3%' (vírgula decimal, como em português)."""
    return f"{v * 100:.{casas}f}%".replace(".", ",")


def fmt_eixo(casas=0):
    return matplotlib.ticker.FuncFormatter(lambda v, _: pct(v, casas))


def fig_curvas(results_dir, path):
    df = pd.read_csv(os.path.join(results_dir, "history.csv"))
    best = df.loc[df["val_acc"].idxmax()]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, col, title in [(axes[0], "loss", "Loss por época"), (axes[1], "acc", "Acurácia por época")]:
        ax.plot(df["epoch"], df[f"train_{col}"], color=BLUE, lw=2, label="treino")
        ax.plot(df["epoch"], df[f"val_{col}"], color=ORANGE, lw=2, label="validação")
        ax.set_title(title, loc="left")
        ax.set_xlabel("Época")
        ax.legend(loc="best")
    axes[1].yaxis.set_major_formatter(fmt_eixo())
    axes[1].scatter([best["epoch"]], [best["val_acc"]], s=70, color=ORANGE,
                    edgecolor=TEXT, linewidth=1.5, zorder=3,
                    label=f"melhor validação: {pct(best['val_acc'])} (época {int(best['epoch'])})")
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_comparacao(resumo_json, path):
    with open(resumo_json, encoding="utf-8") as f:
        versoes = json.load(f)["versoes"]
    names = [v["nome"] for v in versoes]
    vals = [v["val"] for v in versoes]
    tests = [v["teste"] for v in versoes]
    x = range(len(names))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.8))
    b1 = ax.bar([i - w / 2 - 0.01 for i in x], vals, w, color=BLUE, label="validação (melhor época)")
    b2 = ax.bar([i + w / 2 + 0.01 for i in x], [t or 0 for t in tests], w, color=ORANGE, label="teste")
    for bars, values in [(b1, vals), (b2, tests)]:
        for bar, v in zip(bars, values):
            if v:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, pct(v),
                        ha="center", va="bottom", color=TEXT, fontsize=10)
    ax.axhline(0.125, color=TEXT_2, lw=1, ls="--")
    ax.text(-0.45, 0.135, "acaso (12,5%)", ha="left", color=TEXT_2, fontsize=9)
    ax.set_xticks(list(x), names)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(fmt_eixo())
    ax.set_title("Acurácia por versão do modelo", loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_por_genero(results_dir, path):
    df = pd.read_csv(os.path.join(results_dir, "test_per_class.csv")).sort_values("acuracia")
    with open(os.path.join(results_dir, "test_metrics.json"), encoding="utf-8") as f:
        overall = json.load(f)["test_accuracy"]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(df["genero"], df["acuracia"], color=BLUE, height=0.6)
    for y, v in enumerate(df["acuracia"]):
        ax.text(v + 0.01, y, pct(v, 0), va="center", color=TEXT, fontsize=10)
    ax.axvline(overall, color=TEXT_2, lw=1, ls="--")
    ax.text(overall + 0.005, len(df) - 0.45, f"média geral: {pct(overall)}", color=TEXT_2, fontsize=9)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(fmt_eixo())
    ax.set_title("Acurácia no teste por gênero", loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_log(results_dir, path, max_lines=34, command="python train.py"):
    """Renderiza o log real do treino com aparência de terminal."""
    with open(os.path.join(results_dir, "train_log.txt"), encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]
    head = [l for l in lines if not l.startswith("Época")][:4]
    epochs = [l for l in lines if l.startswith("Época")]
    tail = [l for l in lines if not l.startswith("Época")][4:]
    if len(epochs) > max_lines:
        cut = max_lines // 2
        epochs = epochs[:cut] + [f"   ... ({len(epochs) - 2 * cut} épocas omitidas) ..."] + epochs[-cut:]
    shown = [f"PS music-dl> {command}"] + head + epochs + tail

    fig = plt.figure(figsize=(15, 0.26 * len(shown) + 0.6), facecolor="#1e1e1e")
    for i, line in enumerate(shown):
        color = "#d4d4d4"
        if "melhor modelo" in line:
            color = "#6fd08c"
        elif line.startswith(("PS ", "Treino concluído", "Acurácia no teste", "Early")):
            color = "#f0c674"
        fig.text(0.01, 1 - (i + 1) / (len(shown) + 1), line, family="monospace",
                 fontsize=8.6, color=color, va="center")
    fig.savefig(path, dpi=200, facecolor="#1e1e1e")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", required=True, help="pasta de resultados do modelo final")
    parser.add_argument("--command", default="python train.py",
                        help="comando exato que foi usado no treino (aparece no print do log)")
    parser.add_argument("--out", default=OUT_DIR, help="pasta de saída das figuras")
    parser.add_argument("--train-dir", default=None,
                        help="pasta com history.csv/train_log.txt do treino mostrado nas curvas")
    args = parser.parse_args()
    train_dir = args.train_dir or args.final
    out = args.out
    os.makedirs(out, exist_ok=True)

    fig_curvas(train_dir, os.path.join(out, "01_curvas_treino.png"))
    resumo = os.path.join(args.final, "resumo_versoes.json")
    if os.path.exists(resumo):
        fig_comparacao(resumo, os.path.join(out, "02_comparacao_versoes.png"))
    if os.path.exists(os.path.join(args.final, "test_per_class.csv")):
        fig_por_genero(args.final, os.path.join(out, "03_acuracia_por_genero.png"))
        shutil.copy(os.path.join(args.final, "confusion_matrix.png"),
                    os.path.join(out, "04_matriz_confusao.png"))
    if os.path.exists(os.path.join(train_dir, "train_log.txt")):
        fig_log(train_dir, os.path.join(out, "05_log_treino.png"), command=args.command)
    print(f"Figuras salvas em {out}/")


if __name__ == "__main__":
    main()
