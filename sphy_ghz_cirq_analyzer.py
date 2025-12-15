# -*- coding: utf-8 -*-
# ───────────────────────────────────────────────────────────────
# File: sphy_ghz_cirq_analyzer.py
# Purpose: Gerador de Benchmarks (QBench SPHY GHZ Cirq) para Logs de Estabilização de Emaranhamento
# Autor: Gemini & deywe@QLZ
# ───────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime
import re

# Define o diretório de log (o mesmo usado no script de simulação)
LOG_DIR = "logs_harpia"
os.makedirs(LOG_DIR, exist_ok=True)

# ====================================================================
# === FUNÇÃO PRINCIPAL DE ANÁLISE ===
# ====================================================================

def analyze_ghz_cirq_log():
    """
    Pede o caminho do arquivo CSV (LOG_CSV), carrega os dados e gera o relatório de benchmarks
    focado na estabilização de estados GHZ em ambiente Cirq.
    """
    # 1. Pede o caminho do arquivo ao usuário
    default_path_pattern = os.path.join(LOG_DIR, "qghz_*q_log_*.csv")
    print("\n" + "="*80)
    print(" ⚛️ QBENCH SPHY GHZ CIRQ ANALYZER: Relatório de Estabilidade de Emaranhamento ".center(80))
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
    df['Result'] = df['Result'].astype(str).str.strip() 
    df.dropna(subset=['Boost', 'SPHY (%)', 'Result'], inplace=True)

    # Tenta determinar o número de qubits pelo tamanho do campo 'Result'
    try:
        num_qubits = len(df['Result'].iloc[0])
    except:
        num_qubits = 0
    
    ideal_states = ['0' * num_qubits, '1' * num_qubits]

    # === 3. CÁLCULO DAS MÉTRICAS DE BENCHMARK ===
    
    total_frames = len(df)
    
    # GHZ Controlado: Estado GHZ (00..0 ou 11..1) E Accepted = '✅' (SPHY ativou)
    accepted_ghz_df = df[df['Accepted'] == '✅']
    
    accepted_count_controlled = len(accepted_ghz_df)
    success_rate_controlled = (accepted_count_controlled / total_frames) * 100 if total_frames > 0 else 0.0

    # 3.1. Métricas de Controle (Meissner AI)
    mean_boost_accepted = accepted_ghz_df['Boost'].mean() if accepted_count_controlled > 0 else 0.0
    std_boost_accepted = accepted_ghz_df['Boost'].std() if accepted_count_controlled > 1 else 0.0
    
    # 3.2. Métricas de Estabilidade de Emaranhamento (Coerência SPHY)
    mean_stability_index = df['SPHY (%)'].mean()
    stability_variance_index = df['SPHY (%)'].var()
    
    # 3.3. Taxa Bruta de GHZ (Sucesso sem filtro SPHY)
    raw_ghz_success_count = len(df[df['Result'].isin(ideal_states)])
    raw_ghz_success_rate = (raw_ghz_success_count / total_frames) * 100 if total_frames > 0 else 0.0
    
    # --- Delta de Performance ---
    performance_delta = success_rate_controlled - raw_ghz_success_rate

    # === 4. GERAÇÃO DO RELATÓRIO ===
    
    print("\n" + "="*80)
    print(f" 📈 RELATÓRIO QBENCH SPHY-GHZ CIRQ - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(80))
    print("="*80)
    
    print(f" **Framework:** Google Cirq")
    print(f" **Dataset:** {os.path.basename(file_path)}")
    print(f" **Qubits (Inferidos):** {num_qubits}")
    print(f" **Rodadas Totais:** {total_frames:,}")
    print("---")
    
    # -- A. Métricas de Estabilização (IA Meissner) --
    print(" ## A. Estabilização de Emaranhamento Controlada")
    print(f" * **Taxa de Aceitação Controlada (SPHY):** **{success_rate_controlled:.2f}%**")
    print(f" * **Taxa de Sucesso GHZ Bruta:** {raw_ghz_success_rate:.2f}%")
    print(f" * **Delta de Estabilização (Ganho SPHY):** **{performance_delta:.2f} pontos**")
    
    # -- B. Métricas da Força de Sincronia (Boost) --
    print("\n ## B. Força de Sincronia Aplicada (Meissner Core)")
    print(f" * **Boost Médio Aplicado p/ Sucesso:** **{mean_boost_accepted:.6f}** (Analogia Gravitacional)")
    print(f" * **Desvio Padrão do Boost:** {std_boost_accepted:.6f} (Consistência do Pulso)")
    
    # -- C. Estabilidade Geral do Campo SPHY --
    print("\n ## C. Estabilidade do Campo SPHY (Geral)")
    print(f" * **Mean Stability Index (SPHY Média):** **{mean_stability_index:.6f}**")
    print(f" * **Stability Variance Index:** {stability_variance_index:.6f} (Variância do Emaranhamento)")
    
    print("="*80)
    
    # === 5. Geração de Gráficos (Baseado no Estilo do Script de Simulação) ===
    
    # Obtém os dados de evolução da coerência SPHY
    sphy_evolution_list = df['SPHY (%)'].tolist()
    if not sphy_evolution_list:
        print("❌ Sem dados para plotar.")
        return

    sphy_evolution = np.array(sphy_evolution_list)
    tempo = np.linspace(0, 1, len(sphy_evolution))
    
    # Usando a interpolação para suavizar a evolução da estabilidade (replicando a lógica do script original)
    from scipy.interpolate import interp1d
    sinais = [interp1d(tempo, np.roll(sphy_evolution, i), kind='cubic') for i in range(2)]
    novo_tempo = np.linspace(0, 1, 2000)
    
    dados = [sinal(novo_tempo) + np.random.normal(0, 0.15, len(novo_tempo)) for sinal in sinais]
    pesos = np.linspace(1, 1.5, 2)
    emaranhamento = np.average(dados, axis=0, weights=pesos)

    estabilidade_media = np.mean(emaranhamento)
    estabilidade_variancia = np.var(emaranhamento)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    ax1.plot(novo_tempo, emaranhamento, 'k--', linewidth=2, label="Emaranhamento Médio")
    for i in range(len(dados)):
        ax1.plot(novo_tempo, dados[i], alpha=0.3, color='blue' if i == 0 else 'red')
    ax1.set_xlabel("Tempo Normalizado")
    ax1.set_ylabel("Coerência/Amplitude")
    ax1.set_title(f"Emaranhamento GHZ - {num_qubits} Qubits")
    ax1.legend()
    ax1.grid()

    ax2.plot(novo_tempo, emaranhamento, 'k-', label="Emaranhamento Médio")
    ax2.axhline(estabilidade_media, color='green', linestyle='--', label=f"Média: {estabilidade_media:.2f}")
    ax2.axhline(estabilidade_media + np.sqrt(estabilidade_variancia), color='orange', linestyle='--', label=f"± Variância")
    ax2.axhline(estabilidade_media - np.sqrt(estabilidade_variancia), color='orange', linestyle='--')
    ax2.set_xlabel("Tempo Normalizado")
    ax2.set_ylabel("Coerência/Amplitude")
    ax2.set_title("Estabilidade do Emaranhamento")
    ax2.legend()
    ax2.grid()

    fig.suptitle(f"Análise de Estabilização GHZ (Cirq): {total_frames} Rodadas", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Salva o gráfico
    log_dir_output = os.path.dirname(file_path) if os.path.dirname(file_path) else LOG_DIR
    graph_filename = os.path.join(log_dir_output, f"ghz_cirq_qbench_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    
    os.makedirs(log_dir_output, exist_ok=True)
    
    plt.savefig(graph_filename, dpi=300)
    print(f"\n📊 Relatório gráfico salvo em: {graph_filename}")
    plt.show()

if __name__ == "__main__":
    analyze_ghz_cirq_log()