"""
preprocess.py
Converte TODOS os mp3 do FMA Small em Mel Spectrograms UMA ÚNICA VEZ e salva em disco.

Por que isso existe?
    Antes, a cada época o dataset abria 6400 mp3, decodificava o áudio com librosa e
    calculava o espectrograma de novo. Isso é lento (e gera os avisos de
    PySoundFile/audioread). Como o espectrograma de uma música não muda entre épocas,
    calculamos uma vez só aqui e o treino passa a apenas ler números do disco.

O que é gerado na pasta `data_mels/`:
    {split}_mels.npy    -> array (N, 128, 1292) float16 com o Mel Spectrogram em dB
    {split}_labels.npy  -> índice do gênero de cada faixa (int64)
    {split}_ids.npy     -> track_id de cada faixa (para rastrear erros depois)
    {split}_lengths.npy -> quantos frames têm áudio de verdade (o resto é padding)
    {split}_valid.npy   -> False para mp3 que falharam ao carregar (são ignorados no treino)
    norm_stats.npz      -> média e desvio padrão POR FAIXA DE FREQUÊNCIA, calculados só no treino
    meta.json           -> parâmetros usados e a lista de gêneros

Como rodar (da raiz do projeto, com a venv ativada):
    python preprocess.py
Demora alguns minutos (uma vez só). Ocupa ~2,7 GB em disco.
Se for interrompido, rode de novo: splits já finalizados são pulados.
"""

import os
import json
import time
import warnings
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

# ==== Configurações (devem bater com o que o modelo espera) ====
AUDIO_DIR = "fma_small"
TRACKS_CSV = "fma_metadata/tracks.csv"
OUT_DIR = "data_mels"

SAMPLE_RATE = 22050
DURATION = 30                       # segundos
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
# Número de frames de tempo de um áudio de 30s: 1 + (22050*30) // 512 = 1292
N_FRAMES = 1 + (SAMPLE_RATE * DURATION) // HOP_LENGTH
MIN_SECONDS = 5                     # áudios com menos que isso são considerados inválidos

SPLITS = ["training", "validation", "test"]


def track_path(track_id):
    # FMA organiza os arquivos em subpastas pelos 3 primeiros dígitos do ID
    tid = "{:06d}".format(track_id)
    return os.path.join(AUDIO_DIR, tid[:3], tid + ".mp3")


def compute_mel(track_id):
    """
    Roda em um processo separado (paralelo). Retorna (mel_db, n_frames_reais, erro).
    mel_db é None se o arquivo não pôde ser lido.
    """
    # Importamos aqui dentro para cada processo filho ter o seu librosa
    import librosa
    warnings.filterwarnings("ignore")  # silencia avisos de PySoundFile/audioread

    path = track_path(track_id)
    try:
        y, _ = librosa.load(path, sr=SAMPLE_RATE, duration=DURATION)
    except Exception as e:
        return None, 0, f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"

    if len(y) < SAMPLE_RATE * MIN_SECONDS:
        return None, 0, f"áudio curto demais ({len(y) / SAMPLE_RATE:.1f}s)"

    # Quantos frames do espectrograma correspondem a áudio real (antes do padding)
    real_frames = min(N_FRAMES, 1 + len(y) // HOP_LENGTH)

    # Garante duração fixa: completa com silêncio ou corta
    target_len = SAMPLE_RATE * DURATION
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    # ref=1.0 (fixo) em vez de ref=np.max: assim o volume de cada música é
    # medido na MESMA escala para todas, e a normalização é feita depois com as
    # estatísticas do conjunto de treino. top_db=80 corta valores muito baixos.
    mel_db = librosa.power_to_db(mel, ref=1.0, top_db=80.0)
    return mel_db.astype(np.float16), real_frames, None


def load_split_table(split):
    tracks = pd.read_csv(TRACKS_CSV, index_col=0, header=[0, 1])
    small = tracks[tracks["set", "subset"] == "small"]
    genres_all = sorted(small["track", "genre_top"].dropna().unique())
    df = small[(small["set", "split"] == split) & small["track", "genre_top"].notna()]
    return df.index.to_numpy(), df["track", "genre_top"].to_numpy(), genres_all


def process_split(split, workers, limit=None):
    short = {"training": "train", "validation": "val", "test": "test"}[split]
    mels_path = os.path.join(OUT_DIR, f"{short}_mels.npy")
    done_flag = os.path.join(OUT_DIR, f"{short}_DONE")

    if os.path.exists(done_flag):
        print(f"[{split}] já processado, pulando.")
        return

    ids, genre_names, genres_all = load_split_table(split)
    if limit:  # só para testes rápidos do código
        ids, genre_names = ids[:limit], genre_names[:limit]
    genre_to_idx = {g: i for i, g in enumerate(genres_all)}
    labels = np.array([genre_to_idx[g] for g in genre_names], dtype=np.int64)
    n = len(ids)
    print(f"[{split}] {n} faixas -> {mels_path}")

    # open_memmap cria o .npy direto no disco e vamos preenchendo aos poucos,
    # sem precisar manter 2 GB de dados na RAM.
    mels = np.lib.format.open_memmap(
        mels_path, mode="w+", dtype=np.float16, shape=(n, N_MELS, N_FRAMES)
    )
    lengths = np.zeros(n, dtype=np.int64)
    valid = np.zeros(n, dtype=bool)
    errors = []

    start = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        # chunksize > 1 reduz o custo de comunicação entre processos
        results = pool.map(compute_mel, ids.tolist(), chunksize=8)
        for i, (mel, real_frames, err) in enumerate(results):
            if mel is None:
                errors.append((int(ids[i]), err))
                mels[i] = 0
            else:
                mels[i] = mel
                lengths[i] = real_frames
                valid[i] = True
            if (i + 1) % 200 == 0 or i + 1 == n:
                elapsed = time.time() - start
                eta = elapsed / (i + 1) * (n - i - 1)
                print(f"  {i + 1}/{n}  ({elapsed:.0f}s, faltam ~{eta:.0f}s)")

    mels.flush()
    del mels
    np.save(os.path.join(OUT_DIR, f"{short}_labels.npy"), labels)
    np.save(os.path.join(OUT_DIR, f"{short}_ids.npy"), ids.astype(np.int64))
    np.save(os.path.join(OUT_DIR, f"{short}_lengths.npy"), lengths)
    np.save(os.path.join(OUT_DIR, f"{short}_valid.npy"), valid)

    if errors:
        print(f"[aviso] {len(errors)} faixa(s) inválida(s) em {split} (serão ignoradas):")
        for tid, err in errors:
            print(f"    {tid}: {err}")

    # Marca o split como concluído (permite retomar se o script for interrompido)
    open(done_flag, "w").close()


def compute_norm_stats():
    """
    Calcula média e desvio padrão de cada uma das 128 faixas de frequência (mel bins),
    usando SOMENTE o conjunto de treino e somente os frames com áudio real.
    Usar só o treino evita "vazar" informação da validação/teste para o modelo.
    """
    mels = np.load(os.path.join(OUT_DIR, "train_mels.npy"), mmap_mode="r")
    lengths = np.load(os.path.join(OUT_DIR, "train_lengths.npy"))
    valid = np.load(os.path.join(OUT_DIR, "train_valid.npy"))

    total = np.zeros(N_MELS, dtype=np.float64)
    total_sq = np.zeros(N_MELS, dtype=np.float64)
    count = 0
    for i in np.flatnonzero(valid):
        m = mels[i, :, : lengths[i]].astype(np.float64)
        total += m.sum(axis=1)
        total_sq += (m ** 2).sum(axis=1)
        count += m.shape[1]

    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - mean ** 2, 1e-8))
    np.savez(os.path.join(OUT_DIR, "norm_stats.npz"),
             mean=mean.astype(np.float32), std=std.astype(np.float32))
    print(f"Estatísticas de normalização: média ~{mean.mean():.1f} dB, desvio ~{std.mean():.1f} dB")


def main():
    global OUT_DIR
    parser = argparse.ArgumentParser()
    # 4 processos é um bom equilíbrio: acelera bastante sem travar o notebook
    parser.add_argument("--workers", type=int, default=4)
    # Opções para testar o código rapidamente com poucas faixas:
    parser.add_argument("--limit", type=int, default=None, help="processa só N faixas por split")
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()

    OUT_DIR = args.out_dir
    os.makedirs(OUT_DIR, exist_ok=True)
    for split in SPLITS:
        process_split(split, args.workers, args.limit)

    compute_norm_stats()

    _, _, genres_all = load_split_table("training")
    meta = {
        "sample_rate": SAMPLE_RATE, "duration": DURATION, "n_mels": N_MELS,
        "n_fft": N_FFT, "hop_length": HOP_LENGTH, "n_frames": N_FRAMES,
        "genres": list(genres_all),
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print("Pré-processamento concluído.")


if __name__ == "__main__":
    # O "if __name__ == '__main__'" é OBRIGATÓRIO no Windows quando usamos
    # vários processos, senão cada processo filho reexecutaria o script inteiro.
    main()
