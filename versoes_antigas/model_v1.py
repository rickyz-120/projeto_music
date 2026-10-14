"""
model.py
CNN simples para classificar gêneros musicais a partir de Mel Spectrograms.
Entrada esperada: tensor (batch, 1, n_mels, time_steps)
"""

import torch
import torch.nn as nn
import librosa
import numpy as np

class GenreCNN(nn.Module):
    def __init__(self, num_classes, n_mels=128, dropout=0.3):
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    classes_fma = [
        'Electronic', 'Experimental', 'Folk', 'Hip-Hop', 
        'Instrumental', 'International', 'Pop', 'Rock'
    ]
    
    model = GenreCNN(num_classes=8)
    
    # Descomente a linha abaixo para usar a inteligência que você treinou nas 10 épocas
    # model.load_state_dict(torch.load("melhor_modelo.pth"))
    model.eval()

    # Caminho ajustado para um arquivo que seja da classe Rock
    caminho_audio = r"fma_small\135\135054.mp3" 
    genero_verdadeiro = "Rock" 
    
    print(f"Processando o arquivo: {caminho_audio}")
    
    try:
        y, sr = librosa.load(caminho_audio, sr=22050, duration=30.0)
        
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        entrada_real = torch.tensor(mel_spec_db).unsqueeze(0).unsqueeze(0).float()

        with torch.no_grad():
            output = model(entrada_real)
            
        indice_previsto = torch.argmax(output, dim=1).item()
        genero_previsto = classes_fma[indice_previsto]
        
        print("\n" + "="*30)
        print("         RESULTADOS")
        print("="*30)
        print(f"Saída bruta (Logits): \n{output.numpy()}\n")
        
        print(f"➜ Gênero PREVISTO:   {genero_previsto} (Índice {indice_previsto})")
        print(f"➜ Gênero VERDADEIRO: {genero_verdadeiro}\n")
        
        if genero_previsto == genero_verdadeiro:
            print("✅ O modelo ACERTOU!")
        else:
            print("❌ O modelo ERROU.")
            print("(Lembrete: Sem carregar os pesos .pth finais, o modelo escolhe as características aleatoriamente)")
            
        print("="*30)

    except Exception as e:
        print(f"Erro ao carregar o arquivo: {e}")
        print("O arquivo .mp3 não foi encontrado. Substitua 'caminho_audio' por um arquivo Rock válido na sua pasta.")