"""
dataset.py
Dataset customizado para o FMA Small.
Carrega os arquivos de áudio (.mp3), converte para Mel Spectrogram e
retorna (espectrograma, label) para o DataLoader do PyTorch.
"""

import os
import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset


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