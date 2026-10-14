"""
sanity_check.py
Teste inicial da arquitetura CNN para classificacao de generos musicais.
Roda em CPU ou GPU (usa GPU automaticamente se disponivel).

Objetivo desta etapa: validar que a arquitetura da rede processa
corretamente um tensor no formato de um Mel Spectrogram e devolve
a saida esperada (uma classe entre os generos do FMA Small).
"""

import torch
import torch.nn as nn


class GenreCNN(nn.Module):
    def __init__(self, num_classes, dropout=0.3):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2))
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(256, num_classes))

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.global_pool(x)
        return self.classifier(x)


def main():
    print("=" * 55)
    print("SANITY CHECK - CNN para classificacao de generos musicais")
    print("=" * 55)

    # ---- Dispositivo ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDispositivo em uso: {device}")
    if device.type == "cpu":
        print("(GPU nao detectada nesta maquina - normal para esta etapa,")
        print(" o treino completo sera feito em outra maquina com GPU)")

    # ---- Generos do FMA Small ----
    genres = [
        "Electronic", "Experimental", "Folk", "Hip-Hop",
        "Instrumental", "International", "Pop", "Rock",
    ]
    num_classes = len(genres)
    print(f"\nNumero de classes (generos): {num_classes}")
    print(f"Generos: {genres}")

    # ---- Modelo ----
    model = GenreCNN(num_classes=num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModelo instanciado: GenreCNN")
    print(f"Total de parametros treinaveis: {total_params:,}")

    # ---- Forward pass com dado ficticio ----
    # Simula um batch de Mel Spectrograms: (batch, canal, n_mels, time_steps)
    batch_size = 4
    n_mels = 128
    time_steps = 1300
    dummy_input = torch.randn(batch_size, 1, n_mels, time_steps).to(device)

    print(f"\nFormato da entrada (batch ficticio): {tuple(dummy_input.shape)}")

    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Formato da saida: {tuple(output.shape)}")
    print(f"(esperado: ({batch_size}, {num_classes}))")

    predicted_classes = output.argmax(dim=1)
    print(f"\nClasses previstas para o batch ficticio: {predicted_classes.tolist()}")
    print(f"Generos correspondentes: {[genres[i] for i in predicted_classes.tolist()]}")

    print("\n" + "=" * 55)
    print("RESULTADO: modelo executou o forward pass com sucesso.")
    print("=" * 55)


if __name__ == "__main__":
    main()