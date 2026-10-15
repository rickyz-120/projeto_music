"""
predict.py
Classifica o gênero de QUALQUER arquivo de áudio (mp3, wav, ...) com o modelo treinado.

Uso:
    python predict.py musica.mp3
    python predict.py musica1.mp3 musica2.wav --top 3

O modelo vem do arquivo `modelo_v4.pt` (baixe na página de Releases do GitHub e coloque
na pasta do projeto). Esse arquivo é um "pacote": além dos pesos, guarda a arquitetura,
os nomes dos gêneros, a média/desvio da normalização e os parâmetros do Mel Spectrogram.
Por isso o predict.py não precisa do dataset nem da pasta data_mels/.

Para criar o pacote a partir de um checkpoint de treino (só quem treinou precisa fazer):
    python predict.py --exportar checkpoints_v4/best.pt

Como a previsão funciona:
    1. Carrega o áudio a 22050 Hz (igual ao preprocess.py) e calcula o Mel Spectrogram em dB.
    2. Normaliza com a média/desvio do conjunto de TREINO (igual ao dataset.py).
    3. Divide a música em trechos de 30s (o modelo foi avaliado com 30s), classifica
       cada trecho e tira a média das probabilidades.
"""

import os
import sys
import json
import argparse
import warnings

import numpy as np
import torch

from model import GenreCNN

PACOTE = "modelo_v4.pt"
MIN_SEGUNDOS_TRECHO = 10   # trechos finais mais curtos que isso são ignorados
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def exportar(checkpoint_path, data_dir="data_mels", destino=PACOTE):
    """Junta checkpoint + normalização + parâmetros do Mel num único arquivo."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    stats = np.load(os.path.join(data_dir, "norm_stats.npz"))
    with open(os.path.join(data_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    pacote = {
        "model": ckpt["model"],
        "model_kwargs": ckpt["model_kwargs"],
        "genres": meta["genres"],
        "mean": torch.from_numpy(stats["mean"]),
        "std": torch.from_numpy(stats["std"]),
        "mel": {k: meta[k] for k in ("sample_rate", "duration", "n_mels", "n_fft", "hop_length")},
        "origem": {"checkpoint": checkpoint_path, "epoch": ckpt.get("epoch"), "val_acc": ckpt.get("val_acc")},
    }
    torch.save(pacote, destino)
    print(f"Pacote salvo em {destino} ({os.path.getsize(destino) / 1024**2:.1f} MB)")


def carregar_pacote(path=PACOTE):
    if not os.path.exists(path):
        sys.exit(f"Arquivo {path} não encontrado. Baixe-o na página de Releases do repositório "
                 f"e coloque na pasta do projeto.")
    pacote = torch.load(path, map_location="cpu", weights_only=False)
    model = GenreCNN(num_classes=len(pacote["genres"]), **pacote["model_kwargs"])
    model.load_state_dict(pacote["model"])
    return model.to(DEVICE).eval(), pacote


def mel_da_musica(path, mel):
    """Mesmo cálculo do preprocess.py: Mel Spectrogram em dB com ref=1.0 e top_db=80."""
    import librosa
    warnings.filterwarnings("ignore")  # avisos de PySoundFile/audioread ao abrir mp3
    y, _ = librosa.load(path, sr=mel["sample_rate"])
    m = librosa.feature.melspectrogram(y=y, sr=mel["sample_rate"], n_mels=mel["n_mels"],
                                       n_fft=mel["n_fft"], hop_length=mel["hop_length"])
    return librosa.power_to_db(m, ref=1.0, top_db=80.0).astype(np.float32), len(y) / mel["sample_rate"]


@torch.no_grad()
def prever(model, pacote, path):
    mel = pacote["mel"]
    m, segundos = mel_da_musica(path, mel)
    m = (m - pacote["mean"].numpy()[:, None]) / pacote["std"].numpy()[:, None]

    # Trechos de 30s (1292 frames), sem sobreposição
    frames_trecho = 1 + (mel["sample_rate"] * mel["duration"]) // mel["hop_length"]
    min_frames = MIN_SEGUNDOS_TRECHO * mel["sample_rate"] // mel["hop_length"]
    trechos = [m[:, i:i + frames_trecho] for i in range(0, m.shape[1], frames_trecho)]
    trechos = [t for t in trechos if t.shape[1] >= min_frames] or [m]  # áudio curto: usa o que tiver

    probs = []
    for t in trechos:
        x = torch.from_numpy(t).unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1, 1, 128, frames)
        probs.append(torch.softmax(model(x).float(), dim=1)[0].cpu())
    return torch.stack(probs).mean(dim=0), segundos, len(trechos)


def main():
    parser = argparse.ArgumentParser(description="Classifica o gênero musical de arquivos de áudio.")
    parser.add_argument("arquivos", nargs="*", help="um ou mais arquivos de áudio")
    parser.add_argument("--top", type=int, default=3, help="quantos gêneros mostrar")
    parser.add_argument("--modelo", default=PACOTE, help="arquivo do pacote do modelo")
    parser.add_argument("--exportar", metavar="CHECKPOINT",
                        help="cria o pacote a partir de um checkpoint de treino")
    args = parser.parse_args()

    if args.exportar:
        exportar(args.exportar, destino=args.modelo)
        return
    if not args.arquivos:
        parser.error("informe pelo menos um arquivo de áudio")

    model, pacote = carregar_pacote(args.modelo)
    for path in args.arquivos:
        probs, segundos, n = prever(model, pacote, path)
        ordem = probs.argsort(descending=True)[: args.top]
        print(f"\n{os.path.basename(path)}  ({segundos:.0f}s, {n} trecho(s) de 30s)")
        for k, i in enumerate(ordem):
            barra = "#" * int(round(probs[i].item() * 30))
            print(f"  {k + 1}. {pacote['genres'][i]:<14} {probs[i].item():6.1%}  {barra}")


if __name__ == "__main__":
    main()
