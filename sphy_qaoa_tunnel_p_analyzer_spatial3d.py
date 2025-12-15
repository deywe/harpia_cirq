
# file  sphy_qaoa_tunnel_p_analyzer_spatial3d.py
# ...existing code...
# -*- coding: utf-8 -*-
# ───────────────────────────────────────────────────────────────
# File: sphy_qaoa_tunnel_p_analyzer_spatial3d.py
# Purpose: Gerador de Benchmarks (QBench SPHY QAOA c/ P) para Logs de Tunelamento
# Autor: Gemini & deywe@QLZ — Versão remendada com Spatial 3D Plot
# ───────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime
import re

# Define o diretório de log
LOG_DIR = "logs_harpia_sphy_cirq"
os.makedirs(LOG_DIR, exist_ok=True)

# ====================================================================
# === FUNÇÃO PRINCIPAL DE ANÁLISE ===
# ====================================================================

def analyze_qaoa_tunnel_log():
    """
    Pede o caminho do arquivo CSV (LOG_CSV), carrega os dados e gera o relatório de benchmarks,
    focado na comparação entre os modos de entropia 1 e 2. Inclui agora um Spatial Plot 3D
    que visualiza P = [Px,Py,Pz] e Pdot = [Vx,Vy,Vz] extraídos de psi0_noise.
    """
    # 1. Pede o caminho do arquivo ao usuário
    default_path_pattern = os.path.join(LOG_DIR, "harpia_tunnel_cirq_batch_*.csv")
    print("\n" + "="*80)
    print(" ⚛️ QBENCH QAOA TUNNELING ANALYZER: Comparação de Modos Entrópicos ".center(80))
    print("="*80)
    
    file_path = input(f"📁 Digite o caminho completo do LOG_CSV (ex: {default_path_pattern}):\n>> ").strip()
    
    if not os.path.exists(file_path):
        print(f"\n❌ Erro: Arquivo não encontrado em: {file_path}")
        sys.exit(1)
        
    # Extrai o MODE do nome do arquivo para o título
    mode_match = re.search(r"MODE(\d+)", os.path.basename(file_path))
    mode_info = f"MODE {mode_match.group(1)}" if mode_match else "Modo Desconhecido"

    # 2. Carrega o Dataset
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"\n❌ Erro ao carregar o CSV: {e}")
        sys.exit(1)

    # Garante colunas essenciais e realiza limpeza
    required_cols = ['status', 'energy', 'f_opt', 'time', 'psi0_noise']
    if not all(col in df.columns for col in required_cols):
        print(f"\n❌ Erro: Colunas essenciais faltando. Requeridas: {required_cols}")
        sys.exit(1)

    df['f_opt'] = pd.to_numeric(df['f_opt'], errors='coerce')
    df['energy'] = pd.to_numeric(df['energy'], errors='coerce')
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df.dropna(subset=['f_opt', 'energy', 'time'], inplace=True)
    
    # === 3. CÁLCULO DAS MÉTRICAS DE BENCHMARK ===
    
    total_rounds = len(df)
    accepted_df = df[df['status'].str.contains('Accepted', na=False, case=False)].copy()
    accepted_count = len(accepted_df)
    success_rate = (accepted_count / total_rounds) * 100 if total_rounds > 0 else 0.0

    # 3.1. Qualidade da Otimização (Energia QAOA)
    mean_energy = accepted_df['energy'].mean() if accepted_count > 0 else 0.0
    std_energy = accepted_df['energy'].std() if accepted_count > 1 else 0.0
    
    # 3.2. Controle SPHY/Tunelamento (f_opt)
    mean_f_opt = accepted_df['f_opt'].mean() if accepted_count > 0 else 0.0
    variance_f_opt = accepted_df['f_opt'].var() if accepted_count > 1 else 0.0
    
    # 3.3. Tempo e Momento (time, e inferência de P)
    mean_valid_time = accepted_df['time'].mean() if accepted_count > 0 else 0.0
    
    # Função para extrair a Média do Momento (proxy de P)
    def extract_mean_momentum(trace_str):
        try:
            # psi0_noise = "P1;P2;P3;P_dot1;P_dot2;P_dot3"
            parts = [float(x) for x in str(trace_str).split(';')]
            if len(parts) == 6:
                return np.mean(np.abs(parts[3:]))
        except:
            return np.nan
            
    # Cria coluna com proxy de momento no dataframe original (evita SettingWithCopyWarning)
    df['Momentum_Proxy'] = df['psi0_noise'].apply(extract_mean_momentum)
    accepted_df = df[df['status'].str.contains('Accepted', na=False, case=False)].copy()
    mean_momentum_proxy = accepted_df['Momentum_Proxy'].mean() if len(accepted_df) > 0 else 0.0
    
    # === 4. GERAÇÃO DO RELATÓRIO ===
    
    print("\n" + "="*80)
    print(f" 📈 RELATÓRIO QBENCH SPHY QAOA TUNNELING - {mode_info}".center(80))
    print("="*80)
    
    print(f" **Dataset:** {os.path.basename(file_path)}")
    print(f" **Rodadas Totais:** {total_rounds:,}")
    print(f" **Rodadas Aceitas (QAOA Executado):** {accepted_count:,} de {total_rounds:,}")
    print("---")
    
    # -- A. Métricas de Sucesso (Filtro SPHY) --
    print(" ## A. Eficiência do Tunelamento Controlado (SPHY)")
    print(f" * **Taxa de Tunelamento Controlado:** **{success_rate:.2f}%**")
    print(f" * **STDJ (f_opt) Médio (Aceitos):** **{mean_f_opt:.6f}**")
    print(f" * **Variância STDJ (f_opt):** {variance_f_opt:.6f} (Consistência do Gatilho)")
    
    # -- B. Métricas de Qualidade (QAOA) --
    print("\n ## B. Qualidade da Solução Quântica (QAOA)")
    print(f" * **Energia Média QAOA:** **{mean_energy:.6f}** (Alvo Ideal: -1.0)")
    print(f" * **Desvio Padrão da Energia:** {std_energy:.6f} (Confiabilidade da Otimização)")
    
    # -- C. Métricas Dinâmicas (Tempo e Momento P) --
    print("\n ## C. Dinâmica do Campo de Ruído (P e Tempo)")
    print(f" * **Tempo Médio de Gatilho (time):** **{mean_valid_time:.4f}s** (Tempo p/ Atingir Zona de Cancelamento)")
    print(f" * **Momento Médio (Proxy de P):** **{mean_momentum_proxy:.6f}** (Influência da Derivada de Ruído)")
    
    print("="*80)
    
    # === 5. Geração de Gráficos (histogramas) ===
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # Histograma 1: Distribuição da Energia QAOA
    ax[0].hist(accepted_df['energy'], bins=20, color='#00796B', alpha=0.8, edgecolor='black')
    ax[0].axvline(mean_energy, color='red', linestyle='dashed', linewidth=1, label=f'Média: {mean_energy:.6f}')
    ax[0].axvline(-1.0, color='blue', linestyle='dotted', linewidth=1, label='Ideal (-1.0)')
    ax[0].set_title(f'Distribuição da Energia QAOA ({mode_info})')
    ax[0].set_xlabel('Energia (Exp. Value)')
    ax[0].set_ylabel('Frequência')
    ax[0].legend()
    ax[0].grid(axis='y', linestyle='--', alpha=0.5)

    # Histograma 2: Distribuição do Momento (Proxy P)
    ax[1].hist(accepted_df['Momentum_Proxy'], bins=20, color='#E91E63', alpha=0.8, edgecolor='black')
    ax[1].axvline(mean_momentum_proxy, color='darkgreen', linestyle='dashed', linewidth=1, label=f'Média: {mean_momentum_proxy:.6f}')
    ax[1].set_title(f'Distribuição do Momento (Proxy P) - Aceitos ({mode_info})')
    ax[1].set_xlabel('Momento Médio (Proxy de P)')
    ax[1].set_ylabel('Frequência')
    ax[1].legend()
    ax[1].grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Salva o gráfico principal
    log_dir = os.path.dirname(file_path) if os.path.dirname(file_path) else LOG_DIR
    graph_filename = os.path.join(log_dir, f"qaoa_tunnel_p_report_{mode_info.replace(' ', '')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    os.makedirs(log_dir, exist_ok=True)
    plt.savefig(graph_filename, dpi=300)
    print(f"\n📊 Relatório gráfico salvo em: {graph_filename}")
    plt.show()

    # === 6. SPATIAL PLOT DE FASE: ψ(t₀) — Proxy do Campo SPHY ===
    print("\n🌐 Gerando Spatial Plot de Fase...")

    # Extrair P = [Px, Py, Pz] e Pdot = [Vx, Vy, Vz]
    positions = []
    velocities = []

    for row in accepted_df['psi0_noise']:
        try:
            parts = [float(p) for p in str(row).strip().split(';')]
            if len(parts) == 6:
                P = parts[:3]
                Pdot = parts[3:]
                positions.append(P)
                velocities.append(Pdot)
        except:
            continue

    positions = np.array(positions)
    velocities = np.array(velocities)

    if positions.shape[0] == 0:
        print("⚠️ Nenhum vetor válido coletado para o Spatial Plot de Fase.")
    else:
        # Define um número máximo de setas para visualização legível
        max_vectors = 1000
        skip = max(1, positions.shape[0] // max_vectors)  # espacia os dados

        fig_phase = plt.figure(figsize=(10, 8))
        ax3d = fig_phase.add_subplot(111, projection='3d')

        ax3d.quiver(
            positions[::skip, 0], positions[::skip, 1], positions[::skip, 2],
            velocities[::skip, 0], velocities[::skip, 1], velocities[::skip, 2],
            length=0.5, normalize=True, color='deepskyblue', alpha=0.7
        )

        ax3d.set_title(f'Spatial Plot de Fase — Vetores SPHY ({mode_info})')
        ax3d.set_xlabel('P1 (X)')
        ax3d.set_ylabel('P2 (Y)')
        ax3d.set_zlabel('P3 (Z)')
        ax3d.grid(True)

        spatial_plot_filename = os.path.join(log_dir, f"sphy_phase_spatial_plot_{mode_info.replace(' ', '')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.tight_layout()
        plt.savefig(spatial_plot_filename, dpi=300)
        print(f"🌀 Spatial Plot salvo em: {spatial_plot_filename}")

        plt.show()


if __name__ == "__main__":
    analyze_qaoa_tunnel_log()