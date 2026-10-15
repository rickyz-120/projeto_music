"""
panns_cnn14.py  (Parte 2 — Transfer learning)
Arquitetura da CNN14 do PANNs, só para INFERÊNCIA (extrair embeddings).

Fonte: "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition"
(Kong et al., 2020) — https://github.com/qiuqiangkong/audioset_tagging_cnn (licença MIT).
Código adaptado do pacote panns-inference, sem as partes de treino (mixup/SpecAugment),
para não depender do pacote inteiro (ele tenta baixar arquivos com wget ao ser importado).

Repare como a ideia é a mesma da nossa GenreCNN (Parte 1): blocos Conv + BatchNorm + ReLU +
pooling, e pooling global média+máximo no final. A diferença é o tamanho (6 blocos, até 2048
canais, ~80 M parâmetros) e o treino em ~2 milhões de áudios do AudioSet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, pool_size=(2, 2)):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        return F.avg_pool2d(x, kernel_size=pool_size)


class Cnn14(nn.Module):
    def __init__(self, sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64,
                 fmin=50, fmax=14000, classes_num=527):
        super().__init__()
        # O próprio modelo calcula o Mel Spectrogram (log-mel) a partir da onda sonora
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
                                                 win_length=window_size, window="hann", center=True,
                                                 pad_mode="reflect", freeze_parameters=True)
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, n_mels=mel_bins,
                                                 fmin=fmin, fmax=fmax, ref=1.0, amin=1e-10, top_db=None,
                                                 freeze_parameters=True)
        self.bn0 = nn.BatchNorm2d(mel_bins)
        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.conv_block5 = ConvBlock(512, 1024)
        self.conv_block6 = ConvBlock(1024, 2048)
        self.fc1 = nn.Linear(2048, 2048)
        self.fc_audioset = nn.Linear(2048, classes_num)

    def forward(self, waveform):
        """waveform: (batch, amostras) a 32 kHz -> dict com 'embedding' (batch, 2048)."""
        x = self.logmel_extractor(self.spectrogram_extractor(waveform))  # (B, 1, tempo, 64)
        x = self.bn0(x.transpose(1, 3)).transpose(1, 3)
        for block in (self.conv_block1, self.conv_block2, self.conv_block3,
                      self.conv_block4, self.conv_block5):
            x = block(x)
        x = self.conv_block6(x, pool_size=(1, 1))
        x = torch.mean(x, dim=3)                                  # média nas frequências
        x = torch.max(x, dim=2)[0] + torch.mean(x, dim=2)        # máximo + média no tempo
        embedding = F.relu_(self.fc1(x))
        return {"embedding": embedding,
                "clipwise_output": torch.sigmoid(self.fc_audioset(embedding))}
