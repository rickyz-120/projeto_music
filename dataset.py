"""
dataset.py
Datasets para o FMA Small.

- FMASpecDataset (NOVO, usado pelo train.py): lê os espectrogramas já calculados
  pelo preprocess.py, normaliza com as estatísticas do treino e aplica data
  augmentation (recorte aleatório + SpecAugment) apenas no treino.
- FMASmallDataset (ANTIGO, mantido para comparação): carrega o mp3 com librosa
  a cada acesso e calcula o espectrograma na hora (lento).
"""

import os
import json
import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset


SPLIT_SHORT = {"training": "train", "validation": "val", "test": "test"}


class FMASpecDataset(Dataset):
    """
    Lê os Mel Spectrograms pré-processados (pasta data_mels/).

    No TREINO (train=True), cada vez que uma música é pedida:
      1. Sorteamos um trecho de `crop_frames` frames (~10s) dentro dos 30s.
         -> A cada época o modelo vê um pedaço diferente da mesma música,
            o que funciona como "ter mais dados" e reduz overfitting.
      2. Aplicamos SpecAugment: apagamos algumas faixas de frequência e alguns
         intervalos de tempo. -> O modelo não pode depender de um único detalhe
            (ex: só do grave), precisa aprender padrões mais gerais.

    Na VALIDAÇÃO/TESTE (train=False) devolvemos os 30s inteiros, sem augmentation,
    para a avaliação ser sempre a mesma (determinística).
    """

    def __init__(
        self,
        data_dir="data_mels",
        split="training",
        train=False,
        crop_frames=431,          # 431 frames * 512 / 22050 ≈ 10 segundos
        freq_masks=2,             # quantas faixas de frequência apagar
        freq_mask_width=15,       # largura máxima de cada faixa (em mel bins, de 128)
        time_masks=2,             # quantos intervalos de tempo apagar
        time_mask_width=40,       # largura máxima de cada intervalo (~1s)
    ):
        short = SPLIT_SHORT[split]
        self.mels_path = os.path.join(data_dir, f"{short}_mels.npy")
        if not os.path.exists(self.mels_path):
            raise FileNotFoundError(
                f"{self.mels_path} não encontrado. Rode antes: python preprocess.py"
            )

        self.train = train
        self.crop_frames = crop_frames
        self.freq_masks, self.freq_mask_width = freq_masks, freq_mask_width
        self.time_masks, self.time_mask_width = time_masks, time_mask_width

        labels = np.load(os.path.join(data_dir, f"{short}_labels.npy"))
        lengths = np.load(os.path.join(data_dir, f"{short}_lengths.npy"))
        valid = np.load(os.path.join(data_dir, f"{short}_valid.npy"))
        ids = np.load(os.path.join(data_dir, f"{short}_ids.npy"))

        # Guardamos só os índices das faixas válidas (mp3 que carregaram certo)
        self.indices = np.flatnonzero(valid)
        self.labels = labels[self.indices]
        self.lengths = lengths[self.indices]
        self.track_ids = ids[self.indices]

        # Normalização: (valor - média) / desvio, por faixa de frequência.
        # Formato (128, 1) para "broadcast" ao longo do eixo do tempo.
        stats = np.load(os.path.join(data_dir, "norm_stats.npz"))
        self.mean = stats["mean"][:, None].astype(np.float32)
        self.std = stats["std"][:, None].astype(np.float32)

        with open(os.path.join(data_dir, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        self.genres = meta["genres"]
        self.genre_to_idx = {g: i for i, g in enumerate(self.genres)}
        self.idx_to_genre = {i: g for i, g in enumerate(self.genres)}
        self.num_classes = len(self.genres)

        # O arquivo é aberto "preguiçosamente" (só quando o primeiro item é pedido).
        # Isso é importante no Windows: cada worker do DataLoader abre o seu.
        self._mels = None

    def __len__(self):
        return len(self.indices)

    def _get_mels(self):
        if self._mels is None:
            # mmap_mode="r": o arquivo NÃO é carregado inteiro na RAM, cada
            # música é lida do disco só quando necessária.
            self._mels = np.load(self.mels_path, mmap_mode="r")
        return self._mels

    def _random_crop(self, mel, length):
        # Sorteia o início do recorte dentro da parte com áudio real (sem o silêncio
        # de padding do final, se a música tiver menos de 30s).
        max_start = max(0, length - self.crop_frames)
        start = np.random.randint(0, max_start + 1)
        crop = mel[:, start : start + self.crop_frames]
        if crop.shape[1] < self.crop_frames:  # segurança: completa se faltar
            pad = self.crop_frames - crop.shape[1]
            crop = np.pad(crop, ((0, 0), (0, pad)))
        return crop

    def _spec_augment(self, mel):
        # Depois da normalização, 0 é o valor "médio" -> usamos 0 para apagar.
        n_mels, n_frames = mel.shape
        for _ in range(self.freq_masks):
            w = np.random.randint(0, self.freq_mask_width + 1)
            f0 = np.random.randint(0, n_mels - w + 1)
            mel[f0 : f0 + w, :] = 0.0
        for _ in range(self.time_masks):
            w = np.random.randint(0, self.time_mask_width + 1)
            t0 = np.random.randint(0, n_frames - w + 1)
            mel[:, t0 : t0 + w] = 0.0
        return mel

    def __getitem__(self, i):
        idx = self.indices[i]
        mel = self._get_mels()[idx]  # (128, 1292) em dB, float16

        if self.train:
            mel = self._random_crop(mel, int(self.lengths[i]))

        # Converte para float32 e normaliza (cria uma cópia, o arquivo não é alterado)
        mel = (mel.astype(np.float32) - self.mean) / self.std

        if self.train:
            mel = self._spec_augment(mel)

        # (1, n_mels, frames): 1 canal, como uma imagem em tons de cinza
        return torch.from_numpy(mel).unsqueeze(0), int(self.labels[i])


class FMASmallDataset(Dataset):
    def __init__(
        self,
        audio_dir,
        tracks_csv,
        subset="small",
        split="training",
        sample_rate=22050,
        duration=30,
        n_mels=128,
        n_fft=2048,
        hop_length=512,
        transform=None,
    ):
        """
        audio_dir: caminho para a pasta fma_small/ (contém subpastas 000, 001, ...)
        tracks_csv: caminho para fma_metadata/tracks.csv
        subset: 'small' (filtra apenas as faixas do FMA Small)
        split: 'training', 'validation' ou 'test' (usa a coluna do próprio tracks.csv)
        """
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.transform = transform

        # tracks.csv tem cabeçalho em múltiplas linhas (multi-index)
        tracks = pd.read_csv(tracks_csv, index_col=0, header=[0, 1])

        # Filtra apenas o subset "small"
        subset_mask = tracks["set", "subset"] == subset
        split_mask = tracks["set", "split"] == split
        df = tracks[subset_mask & split_mask]

        self.track_ids = df.index.tolist()
        self.genres = df["track", "genre_top"]

        # Remove faixas sem gênero definido
        valid = self.genres.notna()
        self.track_ids = [tid for tid, v in zip(self.track_ids, valid) if v]

        # Mapeia gênero (string) -> índice (int)
        unique_genres = sorted(self.genres.dropna().unique())
        self.genre_to_idx = {g: i for i, g in enumerate(unique_genres)}
        self.idx_to_genre = {i: g for g, i in self.genre_to_idx.items()}
        self.num_classes = len(unique_genres)

    def __len__(self):
        return len(self.track_ids)

    def _track_id_to_path(self, track_id):
        # FMA organiza os arquivos em subpastas pelos 3 primeiros dígitos do ID
        tid_str = "{:06d}".format(track_id)
        return os.path.join(self.audio_dir, tid_str[:3], tid_str + ".mp3")

    def __getitem__(self, idx):
        track_id = self.track_ids[idx]
        path = self._track_id_to_path(track_id)
        label_str = self.genres.loc[track_id]
        label = self.genre_to_idx[label_str]

        try:
            y, sr = librosa.load(
                path, sr=self.sample_rate, duration=self.duration
            )
        except Exception as e:
            # Alguns arquivos do FMA são corrompidos; retorna silêncio nesse caso
            print(f"[aviso] falha ao carregar {path}: {e}")
            y = np.zeros(int(self.sample_rate * self.duration))
            sr = self.sample_rate

        # Garante duração fixa (padding ou corte)
        target_len = self.sample_rate * self.duration
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Normaliza para [0, 1]
        mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)

        # Formato (1, n_mels, time_steps) -> canal único, como uma "imagem"
        mel_tensor = torch.tensor(mel_norm, dtype=torch.float32).unsqueeze(0)

        if self.transform:
            mel_tensor = self.transform(mel_tensor)

        return mel_tensor, label