"""
panns_features.py  (Parte 2 — Transfer learning)
Extrai "embeddings" de cada música com a CNN14 do PANNs, pré-treinada no AudioSet
(~2 milhões de vídeos do YouTube, 527 tipos de som). Roda UMA ÚNICA VEZ.

Ideia do transfer learning:
    A CNN14 já aprendeu a reconhecer timbres, instrumentos, vozes e ritmos ouvindo milhões
    de áudios. Em vez de treinar uma CNN do zero com só 6.400 músicas (Parte 1), usamos a
    CNN14 "congelada" como extratora de características: cada música vira um vetor de
    2048 números (o embedding). Depois, train_transfer.py treina só um classificador
    pequeno em cima desses vetores.

Pré-requisitos:
    pip install torchlibrosa
    pesos em pretrained/Cnn14_mAP=0.431.pth  (https://zenodo.org/record/3987831)
    pasta data_mels/ (usa os mesmos ids/rótulos/faixas válidas da Parte 1)

Saída em data_panns/:
    {split}_emb.npy  -> (N, 4, 2048) float16: [música inteira, trecho 0-10s, 10-20s, 20-30s]
                        (os trechos servem de "data augmentation" no treino da cabeça)
    {split}_labels.npy, {split}_ids.npy, {split}_valid.npy

Uso: python panns_features.py
"""

import os
import time
import warnings
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

AUDIO_DIR = "fma_small"
MELS_DIR = "data_mels"
OUT_DIR = "data_panns"
WEIGHTS = os.path.join("pretrained", "Cnn14_mAP=0.431.pth")
SAMPLE_RATE = 32000        # a CNN14 foi treinada com áudio a 32 kHz
DURATION = 30
SPLITS = {"training": "train", "validation": "val", "test": "test"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_audio(track_id):
    """Roda em processos separados: decodificar mp3 é a parte lenta."""
    import librosa
    warnings.filterwarnings("ignore")
    tid = "{:06d}".format(track_id)
    try:
        y, _ = librosa.load(os.path.join(AUDIO_DIR, tid[:3], tid + ".mp3"),
                            sr=SAMPLE_RATE, duration=DURATION)
    except Exception:
        return None
    target = SAMPLE_RATE * DURATION
    return np.pad(y, (0, max(0, target - len(y))))[:target].astype(np.float32)


def load_cnn14():
    from panns_cnn14 import Cnn14
    model = Cnn14(sample_rate=SAMPLE_RATE, window_size=1024, hop_size=320, mel_bins=64,
                  fmin=50, fmax=14000, classes_num=527)
    ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    return model.to(DEVICE).eval()


@torch.no_grad()
def embed(model, wave):
    """Embedding da música inteira + de 3 trechos de 10s -> (4, 2048)."""
    seg = SAMPLE_RATE * 10
    full = model(torch.from_numpy(wave[None]).to(DEVICE))["embedding"]                  # (1, 2048)
    parts = np.stack([wave[i * seg:(i + 1) * seg] for i in range(3)])                     # (3, 10s)
    parts = model(torch.from_numpy(parts).to(DEVICE))["embedding"]                       # (3, 2048)
    return torch.cat([full, parts]).cpu().numpy().astype(np.float16)


def process_split(model, split, workers):
    short = SPLITS[split]
    out_path = os.path.join(OUT_DIR, f"{short}_emb.npy")
    if os.path.exists(out_path):
        print(f"[{split}] já processado, pulando.")
        return
    ids = np.load(os.path.join(MELS_DIR, f"{short}_ids.npy"))
    labels = np.load(os.path.join(MELS_DIR, f"{short}_labels.npy"))
    valid = np.load(os.path.join(MELS_DIR, f"{short}_valid.npy"))
    n = len(ids)
    emb = np.zeros((n, 4, 2048), dtype=np.float16)
    ok = valid.copy()
    print(f"[{split}] {n} faixas")
    start = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, wave in enumerate(pool.map(load_audio, ids.tolist(), chunksize=4)):
            if wave is None or not valid[i]:
                ok[i] = False
            else:
                emb[i] = embed(model, wave)
            if (i + 1) % 400 == 0 or i + 1 == n:
                el = time.time() - start
                print(f"  {i + 1}/{n}  ({el:.0f}s, faltam ~{el / (i + 1) * (n - i - 1):.0f}s)")
    np.save(os.path.join(OUT_DIR, f"{short}_labels.npy"), labels)
    np.save(os.path.join(OUT_DIR, f"{short}_ids.npy"), ids)
    np.save(os.path.join(OUT_DIR, f"{short}_valid.npy"), ok)
    np.save(out_path, emb)  # salvo por último: se existir, o split está completo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    model = load_cnn14()
    print(f"CNN14 carregada ({sum(p.numel() for p in model.parameters()):,} parâmetros) em {DEVICE}")
    for split in SPLITS:
        process_split(model, split, args.workers)
    print("Embeddings prontos em", OUT_DIR)


if __name__ == "__main__":
    main()
