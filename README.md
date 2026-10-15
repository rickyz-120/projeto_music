# Classificação de Gêneros Musicais com CNN

Projeto da disciplina de Deep Learning (trilha Áudio & Linguagem).
Uma CNN em PyTorch recebe o **Mel Spectrogram** de uma música e classifica em um dos
**8 gêneros** do dataset [FMA Small](https://github.com/mdeff/fma):
Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop e Rock.

## Estrutura

| Arquivo | O que faz |
|---|---|
| `preprocess.py` | Calcula os Mel Spectrograms uma única vez e salva em `data_mels/` |
| `dataset.py` | Lê os espectrogramas, normaliza e aplica data augmentation (recorte + SpecAugment) |
| `model.py` | Arquitetura da CNN (`GenreCNN`) + sanity check (`python model.py`) |
| `train.py` | Treino (AMP, AdamW, cosseno, EMA dos pesos, early stopping, checkpoints) |
| `evaluate.py` | Avaliação no teste ou na validação (`--split`), de um modelo ou ensemble |
| `gerar_figuras.py` | Figuras para a apresentação |
| `versoes_antigas/` | Versões v1 e v2 do código, guardadas para comparação |

## Como rodar

1. Baixe e extraia o dataset (não está no repositório por ser muito pesado):
   - `fma_small.zip` (~7,2 GB) → pasta `fma_small/`
   - `fma_metadata.zip` → pasta `fma_metadata/` (precisa do `tracks.csv`)
   - Links: https://os.unil.cloud.switch.ch/fma/fma_small.zip e https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
2. Instale as dependências: `pip install -r requirements.txt`
   (PyTorch com GPU: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`)
3. Rode a partir da pasta do projeto:

```bash
python preprocess.py   # uma vez só (~20-30 min)
python train.py        # treino
python evaluate.py     # avaliação no teste
```

## Resultados

| Modelo | Validação | Teste |
|---|---|---|
| v2 (5 blocos conv, ~1,1 M parâmetros) | 63,9% | 53,5% |
| v3a (2 conv/bloco, ~2,5 M parâmetros) | 64,3% | 55,0% |
| Ensemble v2 + v3a | 65,6% | 55,0% |
| v4 (v3a + EMA dos pesos, sem mixup) | 66,0% | 54,6% |

Melhores gêneros: Hip-Hop (87%), Electronic (73%), Rock (69%).
Mais difíceis: Experimental (31%), Instrumental (32%), Folk (41%).

### Experimentos da v4 (comparados na validação)

| Experimento | Melhor val | Média das últimas 10 épocas |
|---|---|---|
| v3a (referência) | 64,25% | 62,3% |
| + EMA dos pesos (`--ema 0.999`) | 65,5% | 64,2% |
| + EMA + trecho de 15s (`--crop 646`) | 64,25% | 63,7% |
| **+ EMA, sem mixup (`--mixup 0`)** → v4 | **66,0%** | **64,5%** |
| v4 com 30 épocas | 64,0% | — |
| v4 + conexões residuais (`--residual`) | 65,5% | 64,4% |

A EMA reduziu a oscilação da validação de ±1,6% para ±0,3% por época. No teste, porém,
o resultado ficou empatado com a versão anterior (diferença dentro da margem de ±1,8% de um teste com 800 músicas).
