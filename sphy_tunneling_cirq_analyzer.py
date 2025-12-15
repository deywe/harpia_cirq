# -*- coding: utf-8 -*-
# ───────────────────────────────────────────────────────────────
# File: sphy_tunneling_cirq_analyzer.py
# Purpose: Gerador de Benchmarks (QBench SPHY Tunelamento CIRQ) para Logs de Ressonância de Fase
# Autor: Gemini & deywe@QLZ
# ───────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime
import re

# Define o diretório de log
LOG_DIR = "logs_harpia_tunneling_cirq"
os.makedirs(LOG_DIR, exist_ok=True)

# ====================================================================
# === FUNÇÃO PRINCIPAL DE ANÁLISE ===
# ====================================================================

def analyze_cirq_tunneling_log():
    """
    Pede o caminho do arquivo CSV (LOG_CSV), carrega os dados e gera o relatório de benchmarks
    focado em Tunelamento Controlado e Ressonância de Fase no ambiente Cirq.
    """
    # 1. Pede o caminho do arquivo ao usuário
    default_path_pattern = os.path.join(LOG_DIR, "tunneling_1q_log_*.csv")
    print("\n" + "="*80)
    print(" ⚛️ QBENCH SPHY TUNNELING (CIRQ): Relatório de Ressonância de Fase ".center(80))
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
    required_cols = ['Frame', 'Result', 'H', 'S', 'C', 'I', 'Boost', 'SPHY (%)', 'Accepted']
    if not all(col in df.columns for col in required_cols):
        print(f"\n❌ Erro: Colunas essenciais faltando. Requeridas: {required_cols}")
        sys.exit(1)

    # Limpeza e conversão
    df['Boost'] = pd.to_numeric(df['Boost'], errors='coerce')
    df['SPHY (%)'] = pd.to_numeric(df['SPHY (%)'], errors='coerce')
    df['Result'] = pd.to_numeric(df['Result'], errors='coerce')
    df.dropna(subset=['Boost', 'SPHY (%)', 'Result'], inplace=True)

    # === 3. CÁLCULO DAS MÉTRICAS DE BENCHMARK ===
    
    total_frames = len(df)
    
    # Tunelamento SPHY Controlado: Result = 1 E Accepted = '✅'
    accepted_df = df[(df['Accepted'] == '✅') & (df['Result'] == 1)]
    
    accepted_count = len(accepted_df)
    success_rate = (accepted_count / total_frames) * 100 if total_frames > 0 else 0.0

    # 3.1. Métricas de Tunelamento (Foco no Boost)
    mean_boost_accepted = accepted_df['Boost'].mean() if accepted_count > 0 else 0.0
    std_boost_accepted = accepted_df['Boost'].std() if accepted_count > 1 else 0.0
    
    # 3.2. Métricas de Estabilidade e Variância
    mean_stability_index = df['SPHY (%)'].mean()
    stability_variance_index = df['SPHY (%)'].var()
    
    # 3.3. Ressonância de Fase (Índice de Interação I = |H-S|)
    mean_phase_impact = accepted_df['I'].mean() if accepted_count > 0 else 0.0
    
    # --- Cálculo Adicional: Tunelamento Não Controlado (Apenas sucesso do Qubit) ---
    raw_success_count = len(df[df['Result'] == 1])
    raw_success_rate = (raw_success_count / total_frames) * 100 if total_frames > 0 else 0.0
    
    # --- Delta de Performance ---
    performance_delta = success_rate - raw_success_rate

    # === 4. GERAÇÃO DO RELATÓRIO ===
    
    print("\n" + "="*80)
    print(f" 📈 RELATÓRIO QBENCH SPHY-TUNNELING (CIRQ) - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(80))
    print("="*80)
    
    print(f" **Framework:** Google Cirq")
    print(f" **Dataset:** {os.path.basename(file_path)}")
    print(f" **Tentativas Totais (Frames):** {total_frames:,}")
    print(f" **Sucesso de Tunelamento SPHY Controlado:** {accepted_count:,} de {total_frames:,}")
    print("---")
    
    # -- A. Métricas de Tunelamento Controlado (Efetividade da IA) --
    print(" ## A. Tunelamento Quântico Controlado (SPHY/Meissner)")
    print(f" * **Taxa de Sucesso (Controlado):** **{success_rate:.2f}%**")
    print(f" * **Sucesso NÃO Controlado (Qubit '1' bruto):** {raw_success_rate:.2f}%")
    print(f" * **Delta de Performance SPHY:** **{performance_delta:.2f} pontos** (Ganho de Eficiência)")
    
    # -- B. Métricas da Força de Ressonância (Meissner Core) --
    print("\n ## B. Força de Ressonância de Fase (Meissner Core)")
    print(f" * **Boost Médio (Força SPHY) p/ Sucesso:** **{mean_boost_accepted:.6f}**")
    print(f" * **Impacto Médio da Fase (|H-S|) p/ Sucesso:** **{mean_phase_impact:.6f}** (Analogia à Ressonância)")
    print(f" * **Desvio Padrão do Boost (Consistência):** {std_boost_accepted:.6f}")
    
    # -- C. Estabilidade do Campo SPHY --
    print("\n ## C. Estabilidade do Campo SPHY (Geral)")
    print(f" * **Mean Stability Index (SPHY Média):** **{mean_stability_index:.6f}**")
    print(f" * **Stability Variance Index:** {stability_variance_index:.6f}")
    
    print("="*80)
    
    # === 5. Geração de Histograma ===
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # Histograma 1: Distribuição do Boost (Força SPHY)
    ax[0].hist(accepted_df['Boost'], bins=20, color='#00796B', alpha=0.8, edgecolor='black')
    ax[0].axvline(mean_boost_accepted, color='red', linestyle='dashed', linewidth=1, label=f'Média: {mean_boost_accepted:.6f}')
    ax[0].set_title('Distribuição do Boost (Força SPHY) - Sucessos Controlados')
    ax[0].set_xlabel('Boost (Força de Sincronia $F_{opt}$)')
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
    graph_filename = os.path.join(log_dir, f"tunneling_cirq_qbench_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    
    os.makedirs(log_dir, exist_ok=True)
    
    plt.savefig(graph_filename, dpi=300)
    print(f"\n📊 Relatório gráfico salvo em: {graph_filename}")
    plt.show()

if __name__ == "__main__":
    analyze_cirq_tunneling_log()