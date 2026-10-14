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
| `train.py` | Treino (AMP, AdamW, cosseno, mixup, early stopping, checkpoints) |
| `evaluate.py` | Avaliação no conjunto de teste (um modelo ou ensemble) |
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
| **Ensemble v2 + v3a** | **65,6%** | **55,0%** |

Melhores gêneros: Hip-Hop (87%), Electronic (74%), Rock (65%).
Mais difíceis: Experimental (33%), Folk (37%), Pop (42%).
