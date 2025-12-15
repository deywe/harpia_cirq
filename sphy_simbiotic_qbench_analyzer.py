# -*- coding: utf-8 -*-
# ───────────────────────────────────────────────────────────────
# File: sphy_simbiotic_qbench_analyzer.py
# Purpose: Gerador de Benchmarks (QBench SPHY Simbiotic) para Logs de Estabilizacao GHZ
# Autor: Gemini & deywe@QLZ
# ───────────────────────────────────────────────────────────────
import warnings
# Suprime warnings do PennyLane, comum em simulações
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib
# Configura o backend do Matplotlib para evitar problemas em ambientes sem display gráfico
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

# Define o diretório de log
LOG_DIR = "logs_harpia"
os.makedirs(LOG_DIR, exist_ok=True)

# ====================================================================
# === FUNÇÃO PRINCIPAL DE ANÁLISE ===
# ====================================================================

def analyze_simbiotic_log():
    """
    Pede o caminho do arquivo CSV (LOG_CSV), carrega os dados e gera o relatório de benchmarks
    focado na estabilidade GHZ controlada pelo SPHY Simbiótico.
    """
    # 1. Pede o caminho do arquivo ao usuário
    default_path_pattern = os.path.join(LOG_DIR, "qghz_*q_log_*.csv")
    print("\n" + "="*80)
    print(" 📊 QBENCH SPHY SIMBIOTIC ANALYZER: Relatório de Estabilidade GHZ ".center(80))
    print("="*80)
    
    file_path = input(f"📁 Digite o caminho completo do LOG_CSV (ex: {default_path_pattern}):\n>> ").strip()
    
    if not os.path.exists(file_path):
        print(f"\n❌ Erro: Arquivo não encontrado em: {file_path}")
        sys.exit(1)

    # 2. Carrega o Dataset
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"\n❌ Erro ao carregar o CSV: {e}")
        sys.exit(1)

    # Garante que as colunas essenciais estão presentes
    required_cols = ['Frame', 'Accepted', 'C', 'Boost', 'SPHY (%)', 'I']
    if not all(col in df.columns for col in required_cols):
        print(f"\n❌ Erro: Colunas essenciais faltando. Requeridas: {required_cols}")
        sys.exit(1)

    # Limpeza e conversão
    df['Boost'] = pd.to_numeric(df['Boost'], errors='coerce')
    df['SPHY (%)'] = pd.to_numeric(df['SPHY (%)'], errors='coerce')
    df['C'] = pd.to_numeric(df['C'], errors='coerce')
    df.dropna(subset=['Boost', 'SPHY (%)', 'C'], inplace=True)

    # === 3. CÁLCULO DAS MÉTRICAS DE BENCHMARK ===
    
    total_frames = len(df)
    
    # Filtra os estados que foram 'Accepted' (GHZ estado ideal E boost ativado)
    accepted_df = df[df['Accepted'] == '✅']
    rejected_df = df[df['Accepted'] == '❌']
    
    accepted_count = len(accepted_df)
    success_rate = (accepted_count / total_frames) * 100 if total_frames > 0 else 0.0

    # 3.1. Métricas de Estabilidade (Mean Stability Index / Variance)
    
    # Coerência Base (C) e Coerência Final (SPHY (%))
    baseline_coherence_mean = df['C'].mean() * 100 
    mean_stability_index = df['SPHY (%)'].mean()
    stability_variance_index = df['SPHY (%)'].var()
    
    # Ganho liquido: (SPHY média - Coerência Base média)
    coherence_net_gain = mean_stability_index - baseline_coherence_mean
    
    # 3.2. Métricas de Controle (Boost)
    mean_boost_accepted = accepted_df['Boost'].mean() if accepted_count > 0 else 0.0
    std_boost_accepted = accepted_df['Boost'].std() if accepted_count > 1 else 0.0
    
    # 3.3. Interação (I)
    mean_interaction_index = df['I'].mean()
    
    # === 4. GERAÇÃO DO RELATÓRIO ===
    
    print("\n" + "="*80)
    print(f" 📈 RELATÓRIO QBENCH SPHY SIMBIOTIC - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(80))
    print("="*80)
    
    print(f" **Dataset:** {os.path.basename(file_path)}")
    print(f" **Rodadas Totais (Frames):** {total_frames:,}")
    print("---")
    
    # -- A. Métricas de Sucesso do Filtro Simbiótico --
    print(" ## A. Controle de Sucesso (Filtro HARPIA Simbiótico)")
    print(f" * **Taxa de Aceitação Final (GHZ + SPHY):** **{success_rate:.2f}%**")
    
    # -- B. Métricas de Estabilidade e Coerência (Core) --
    print("\n ## B. Performance de Estabilidade SPHY")
    print(f" * **Coerência Baseline Média (C):** {baseline_coherence_mean:.4f}%")
    print(f" * **Mean Stability Index (SPHY Final):** **{mean_stability_index:.4f}%**")
    print(f" * **Ganho Líquido de Estabilidade:** **{coherence_net_gain:.4f} pontos**")
    print(f" * **Stability Variance Index:** {stability_variance_index:.8f} (Consistência da Estabilização)")
    
    # -- C. Métricas de Ação do Campo Simbiótico (Boost) --
    print("\n ## C. Ação do Campo de Sincronia (Boost)")
    print(f" * **Boost Médio (Aceitos):** **{mean_boost_accepted:.6f}** (Força de Sincronia Aplicada)")
    print(f" * **Boost Desvio Padrão (Consistência):** {std_boost_accepted:.6f}")
    print(f" * **Índice de Interação Média (I=|H-S|):** {mean_interaction_index:.6f}")
    
    print("="*80)
    
    # === 5. Geração de Gráficos ===
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # Histograma 1: Distribuição do Boost
    ax[0].hist(accepted_df['Boost'], bins=30, color='darkcyan', alpha=0.8, edgecolor='black')
    ax[0].axvline(mean_boost_accepted, color='red', linestyle='dashed', linewidth=1, label=f'Média Aceito: {mean_boost_accepted:.6f}')
    ax[0].set_title('Distribuição do Boost (Força de Sincronia) - Estados Aceitos ✅')
    ax[0].set_xlabel('Valor do Boost (F_opt)')
    ax[0].set_ylabel('Frequência')
    ax[0].legend()
    ax[0].grid(axis='y', linestyle='--', alpha=0.5)

    # Gráfico 2: Evolução da Estabilidade (SPHY (%))
    window = min(total_frames, 1000)
    frames = df['Frame'].tail(window)
    coherence = df['SPHY (%)'].tail(window)

    ax[1].plot(frames, coherence, color='#E91E63', linewidth=2)
    ax[1].set_title(f'Evolução da Estabilidade SPHY (Últimos {window} Frames)')
    ax[1].set_xlabel('Frame')
    ax[1].set_ylabel('Mean Stability Index (%)')
    ax[1].axhline(coherence.mean(), color='darkgreen', linestyle='dashed', linewidth=1, label=f'Média: {coherence.mean():.2f}%')
    ax[1].legend()
    ax[1].grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Salva o gráfico
    log_dir = os.path.dirname(file_path) if os.path.dirname(file_path) else LOG_DIR
    graph_filename = os.path.join(log_dir, f"simbiotic_qbench_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    
    os.makedirs(log_dir, exist_ok=True)
    
    plt.savefig(graph_filename, dpi=300)
    print(f"\n📊 Relatório gráfico salvo em: {graph_filename}")
    plt.show()

if __name__ == "__main__":
    analyze_simbiotic_log()