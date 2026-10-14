import os
import glob
import torch
import torch.nn as nn
import librosa
import numpy as np
import pandas as pd
import datetime

# =====================================================================
# 1. ARQUITETURA DO MODELO
# =====================================================================
class GenreCNN(nn.Module):
    def __init__(self, num_classes, n_mels=128, dropout=0.3):
        super().__init__()
        self.conv_block1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2))
        self.conv_block2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.conv_block3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.conv_block4 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, num_classes))

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.global_pool(x)
        return self.classifier(x)

if __name__ == "__main__":
    classes_fma = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 'Pop', 'Rock']
    
    # =====================================================================
    # 2. INICIALIZAÇÃO DO MODELO
    # =====================================================================
    model = GenreCNN(num_classes=8)
    
    # IMPORTANTE: Garanta que esta linha está descomentada e com o nome certo do seu ficheiro .pth
    # model.load_state_dict(torch.load("melhor_modelo.pth"))
    
    model.eval()

    # =====================================================================
    # 3. CARREGAMENTO DOS DADOS REAIS DO GABARITO (tracks.csv)
    # =====================================================================
    caminho_csv = r"fma_metadata\tracks.csv" 
    
    print("A iniciar leitura dos metadados...")
    mapa_generos_verdadeiros = {}
    
    try:
        df_labels = pd.read_csv(caminho_csv, index_col=0, header=[0, 1])
        generos = df_labels[('track', 'genre_top')].dropna()
        mapa_generos_verdadeiros = {str(idx).zfill(6): genero for idx, genero in generos.items()}
        print(f"✅ Gabarito original carregado! {len(mapa_generos_verdadeiros)} faixas mapeadas.")
    except Exception:
        try:
            df_labels = pd.read_csv(caminho_csv)
            col_id = 'track_id' if 'track_id' in df_labels.columns else df_labels.columns[0]
            col_gen = 'genre_top' if 'genre_top' in df_labels.columns else 'genre'
            mapa_generos_verdadeiros = dict(zip(df_labels[col_id].astype(str).str.zfill(6), df_labels[col_gen]))
            print(f"✅ Gabarito simples carregado! {len(mapa_generos_verdadeiros)} faixas mapeadas.")
        except Exception as e:
            print(f"❌ ERRO CRÍTICO ao ler o CSV: {e}")
            exit()

    # =====================================================================
    # 4. PROCESSAMENTO DOS ÁUDIOS EM LOTE
    # =====================================================================
    pasta_dataset = "fma_small"
    arquivos_mp3 = glob.glob(os.path.join(pasta_dataset, "**", "*.mp3"), recursive=True)
    
    # DICA: Para testar rápido, descomente a linha abaixo e processe apenas 20 áudios primeiro.
    # arquivos_mp3 = arquivos_mp3[:20]

    print(f"\nTotal de ficheiros .mp3 encontrados para processamento: {len(arquivos_mp3)}")
    print("A iniciar inferência da rede neuronal...\n")

    resultados = []

    for idx, caminho_audio in enumerate(arquivos_mp3, 1):
        nome_arquivo = os.path.basename(caminho_audio)
        track_id = nome_arquivo.replace(".mp3", "")
        genero_verdadeiro = mapa_generos_verdadeiros.get(track_id, "Desconhecido")
        
        try:
            # Carregamento e extração de características
            y, sr = librosa.load(caminho_audio, sr=22050, duration=30.0)
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # CORREÇÃO DEFINITIVA: Standardização (Z-score) para manter a média em 0 e desvio padrão em 1
            # Isso reflete o comportamento de funções como torch.randn e estabiliza a predição.
            mel_spec_norm = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)

            # Transformação para Tensor
            entrada = torch.tensor(mel_spec_norm).unsqueeze(0).unsqueeze(0).float()

            with torch.no_grad():
                output = model(entrada)
                
            indice_previsto = torch.argmax(output, dim=1).item()
            genero_previsto = classes_fma[indice_previsto]
            
            if genero_verdadeiro == "Desconhecido":
                acertou = "-"
            else:
                acertou = "Sim" if genero_previsto == genero_verdadeiro else "Não"

            resultados.append({
                "Track ID": track_id,
                "Caminho do Ficheiro": caminho_audio,
                "Gênero Previsto (CNN)": genero_previsto,
                "Gênero Verdadeiro (CSV)": genero_verdadeiro,
                "Acertou?": acertou,
                "Status": "Sucesso"
            })

        except Exception as e:
            erro_msg = str(e).strip().split('\n')[0] 
            resultados.append({
                "Track ID": track_id,
                "Caminho do Ficheiro": caminho_audio,
                "Gênero Previsto (CNN)": "-",
                "Gênero Verdadeiro (CSV)": genero_verdadeiro,
                "Acertou?": "-",
                "Status": f"Erro de Leitura: {erro_msg}"
            })
            
        # Imprime o progresso a cada 100 ficheiros
        if idx % 100 == 0 or idx == len(arquivos_mp3):
            print(f"➜ Progresso: {idx} de {len(arquivos_mp3)} processados...")

    # =====================================================================
    # 5. GERAÇÃO DO RELATÓRIO EXCEL (COM NOME DINÂMICO)
    # =====================================================================
    print("\nA exportar relatório para Excel...")
    df_resultados = pd.DataFrame(resultados)
    
    # CORREÇÃO: Gera um nome único com data e hora para evitar o Erro 13 (Permission denied)
    data_hora = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    nome_excel = f"relatorio_modelo_{data_hora}.xlsx"
    
    df_resultados.to_excel(nome_excel, index=False)
    
    total_sucesso = len(df_resultados[df_resultados["Status"] == "Sucesso"])
    total_erros = len(df_resultados[df_resultados["Status"].str.contains("Erro")])
    total_acertos = len(df_resultados[df_resultados["Acertou?"] == "Sim"])
    taxa_acerto = (total_acertos / total_sucesso * 100) if total_sucesso > 0 else 0
    
    print("\n" + "="*45)
    print("               RESUMO FINAL")
    print("="*45)
    print(f"Total de faixas analisadas:  {len(arquivos_mp3)}")
    print(f"Faixas processadas com êxito:{total_sucesso}")
    print(f"Faixas com erro de áudio:    {total_erros} (Listadas no Excel)")
    print(f"Total de acertos exatos:     {total_acertos}")
    print(f"Taxa de acerto real:         {taxa_acerto:.2f}% (ignorando ficheiros com erro)")
    print("="*45)
    print(f"✅ Guardado com sucesso em: '{nome_excel}'")