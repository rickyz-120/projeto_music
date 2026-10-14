"""
model.py  (versão 2 — a versão antiga está em model_v1.py)
CNN para classificar gêneros musicais a partir de Mel Spectrograms.
Entrada esperada: tensor (batch, 1, n_mels, time_steps)

Mudanças em relação à v1:
  - 5 blocos convolucionais em vez de 4, com mais filtros (32→64→128→256→256).
    Mais filtros = mais "detectores de padrões" (timbres, ritmos, harmonias).
  - Pooling GLOBAL no final (média + máximo sobre frequência e tempo) no lugar de
    AdaptiveAvgPool2d((4, 4)). Assim o modelo funciona igual com qualquer
    duração de entrada: treinamos com trechos de ~10s e avaliamos com os 30s.
    Também reduz muito o número de pesos do classificador (menos overfitting).
  - Dropout2d leve nos blocos finais (regularização extra nas convoluções).
  - Conv sem bias quando seguida de BatchNorm (o BatchNorm já tem o seu "bias").
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """(Conv 3x3 -> BatchNorm -> ReLU) x n_convs -> MaxPool 2x2 (-> Dropout2d opcional).

    n_convs=2 é o estilo "VGG": duas convoluções antes de reduzir a resolução,
    o que deixa o bloco enxergar padrões mais complexos (mais capacidade).
    """

    def __init__(self, in_ch, out_ch, drop=0.0, n_convs=1):
        super().__init__()
        layers = []
        for k in range(n_convs):
            layers += [
                nn.Conv2d(in_ch if k == 0 else out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
        layers += [
            nn.MaxPool2d(2),
            # Dropout2d "desliga" canais (mapas de features) inteiros aleatoriamente
            nn.Dropout2d(drop) if drop > 0 else nn.Identity(),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class GenreCNN(nn.Module):
    def __init__(self, num_classes, n_mels=128, dropout=0.4,
                 channels=(32, 64, 128, 256, 256), conv_dropout=0.1, convs_per_block=1):
        super().__init__()

        blocks = []
        in_ch = 1
        for i, out_ch in enumerate(channels):
            # Dropout2d só nos 2 últimos blocos (onde as features são mais "abstratas")
            drop = conv_dropout if i >= len(channels) - 2 else 0.0
            blocks.append(ConvBlock(in_ch, out_ch, drop, n_convs=convs_per_block))
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)

        # Pooling global: resume cada mapa de features em 1 número.
        # Média = "quanto esse padrão aparece no geral"
        # Máximo = "esse padrão aparece forte em algum momento?"
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_ch * 2, 256),   # *2 porque concatenamos média e máximo
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),  # 8 logits (softmax fica dentro da loss)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.cat([self.avg_pool(x), self.max_pool(x)], dim=1)
        return self.classifier(x)


def predict_logits(model, x, crop_frames=431, hop_frames=215):
    """
    Avaliação "multi-recorte": em vez de mostrar os 30s de uma vez, divide a música
    em trechos de ~10s (o MESMO tamanho usado no treino) com 50% de sobreposição,
    classifica cada trecho e tira a média das probabilidades.
    É como pedir a opinião de vários "jurados" sobre partes diferentes da música.

    x: (batch, 1, n_mels, frames_totais)  ->  retorna log-probabilidades (batch, n_classes)
    Se crop_frames=None, usa a música inteira de uma vez (comportamento da v2).
    """
    if crop_frames is None or x.shape[-1] <= crop_frames:
        return torch.log_softmax(model(x).float(), dim=1)
    b = x.shape[0]
    # unfold recorta janelas ao longo do eixo do tempo: (B, 1, mels, n_janelas, crop)
    crops = x.unfold(3, crop_frames, hop_frames)
    n = crops.shape[3]
    crops = crops.permute(0, 3, 1, 2, 4).reshape(b * n, 1, x.shape[2], crop_frames)
    probs = torch.softmax(model(crops).float(), dim=1).view(b, n, -1).mean(dim=1)
    return torch.log(probs + 1e-8)


if __name__ == "__main__":
    # Sanity check: forward + backward com dados fictícios, usando mixed precision,
    # e mede o pico de memória da GPU. Não precisa do dataset.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    model = GenreCNN(num_classes=8).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parâmetros treináveis: {n_params:,}")

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Treino: batch 32 com recortes de ~10s (431 frames)
    if use_amp:
        torch.cuda.reset_peak_memory_stats()
    model.train()
    x = torch.randn(32, 1, 128, 431, device=device)
    y = torch.randint(0, 8, (32,), device=device)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        out = model(x)
        loss = nn.functional.cross_entropy(out, y)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    print(f"Treino   entrada {tuple(x.shape)} -> saída {tuple(out.shape)}, loss={loss.item():.3f}")
    if use_amp:
        print(f"  pico de VRAM no treino: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")

    # Avaliação: batch 16 com a música inteira (30s = 1292 frames)
    if use_amp:
        torch.cuda.reset_peak_memory_stats()
    model.eval()
    x = torch.randn(16, 1, 128, 1292, device=device)
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        out = model(x)
    print(f"Avaliação entrada {tuple(x.shape)} -> saída {tuple(out.shape)}")
    if use_amp:
        print(f"  pico de VRAM na avaliação: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")
    print("OK: arquitetura funcionando.")
