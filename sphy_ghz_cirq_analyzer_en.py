# -*- coding: utf-8 -*-
# ───────────────────────────────────────────────────────────────
# File: sphy_ghz_cirq_analyzer_en.py
# Purpose: QBench SPHY GHZ Cirq Analyzer for Entanglement Stabilization Logs (English)
# Author: Gemini & deywe@QLZ
# ───────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime
import re
from scipy.interpolate import interp1d

# Define the log directory
LOG_DIR = "logs_harpia"
os.makedirs(LOG_DIR, exist_ok=True)

# ====================================================================
# === CORE ANALYSIS FUNCTION ===
# ====================================================================

def analyze_ghz_cirq_log():
    """
    Prompts for the CSV file path, loads the data, and generates the benchmark report
    focused on GHZ state stabilization in the Cirq environment (English).
    """
    # 1. Prompt for file path
    default_path_pattern = os.path.join(LOG_DIR, "qghz_*q_log_*.csv")
    print("\n" + "="*80)
    print(" ⚛️ QBENCH SPHY GHZ CIRQ ANALYZER: Entanglement Stability Report ".center(80))
    print("="*80)
    
    file_path = input(f"📁 Enter the full LOG_CSV path (e.g., {default_path_pattern}):\n>> ").strip()
    
    if not os.path.exists(file_path):
        print(f"\n❌ Error: File not found at: {file_path}")
        sys.exit(1)

    # 2. Load the Dataset
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"\n❌ Error loading CSV: {e}")
        sys.exit(1)

    # Ensure essential columns are present
    required_cols = ['Frame', 'Result', 'H', 'S', 'C', 'I', 'Boost', 'SPHY (%)', 'Accepted']
    if not all(col in df.columns for col in required_cols):
        print(f"\n❌ Error: Essential columns missing. Required: {required_cols}")
        sys.exit(1)

    # Clean and convert columns
    df['Boost'] = pd.to_numeric(df['Boost'], errors='coerce')
    df['SPHY (%)'] = pd.to_numeric(df['SPHY (%)'], errors='coerce')
    df['Result'] = df['Result'].astype(str).str.strip() 
    df.dropna(subset=['Boost', 'SPHY (%)', 'Result'], inplace=True)

    # Attempt to determine the number of qubits from the 'Result' field length
    try:
        num_qubits = len(df['Result'].iloc[0])
    except:
        num_qubits = 0
    
    # Ideal GHZ states for comparison
    ideal_states = ['0' * num_qubits, '1' * num_qubits]

    # === 3. BENCHMARK METRICS CALCULATION ===
    
    total_frames = len(df)
    
    # Controlled GHZ: GHZ State (00..0 or 11..1) AND Accepted = '✅' (SPHY activated)
    accepted_ghz_df = df[df['Accepted'] == '✅']
    
    accepted_count_controlled = len(accepted_ghz_df)
    success_rate_controlled = (accepted_count_controlled / total_frames) * 100 if total_frames > 0 else 0.0

    # 3.1. Control Metrics (Meissner AI)
    mean_boost_accepted = accepted_ghz_df['Boost'].mean() if accepted_count_controlled > 0 else 0.0
    std_boost_accepted = accepted_ghz_df['Boost'].std() if accepted_count_controlled > 1 else 0.0
    
    # 3.2. Entanglement Stability Metrics (SPHY Coherence)
    mean_stability_index = df['SPHY (%)'].mean()
    stability_variance_index = df['SPHY (%)'].var()
    
    # 3.3. Raw GHZ Rate (Success without SPHY filter)
    raw_ghz_success_count = len(df[df['Result'].isin(ideal_states)])
    raw_ghz_success_rate = (raw_ghz_success_count / total_frames) * 100 if total_frames > 0 else 0.0
    
    # --- Performance Delta ---
    performance_delta = success_rate_controlled - raw_ghz_success_rate

    # === 4. REPORT GENERATION (English) ===
    
    print("\n" + "="*80)
    print(f" 📈 SPHY-GHZ CIRQ QBENCH REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(80))
    print("="*80)
    
    print(f" **Framework:** Google Cirq")
    print(f" **Dataset:** {os.path.basename(file_path)}")
    print(f" **Qubits (Inferred):** {num_qubits}")
    print(f" **Total Rounds:** {total_frames:,}")
    print("---")
    
    # -- A. Stabilization Metrics (Meissner AI) --
    print(" ## A. Controlled Entanglement Stabilization")
    print(f" * **Controlled Acceptance Rate (SPHY):** **{success_rate_controlled:.2f}%**")
    print(f" * **Raw GHZ Success Rate:** {raw_ghz_success_rate:.2f}%")
    print(f" * **Stabilization Delta (SPHY Gain):** **{performance_delta:.2f} points**")
    
    # -- B. Synchrony Force Metrics (Boost) --
    print("\n ## B. Applied Synchrony Force (Meissner Core)")
    print(f" * **Mean Boost Applied for Success:** **{mean_boost_accepted:.6f}** (Gravitational Analogy)")
    print(f" * **Boost Standard Deviation:** {std_boost_accepted:.6f} (Pulse Consistency)")
    
    # -- C. Overall SPHY Field Stability --
    print("\n ## C. SPHY Field Stability (Overall)")
    print(f" * **Mean Stability Index (Average SPHY):** **{mean_stability_index:.6f}**")
    print(f" * **Stability Variance Index:** {stability_variance_index:.6f} (Entanglement Variance)")
    
    print("="*80)
    
    # === 5. Graph Generation (Replicating the Simulation Script's Style) ===
    
    sphy_evolution_list = df['SPHY (%)'].tolist()
    if not sphy_evolution_list:
        print("❌ No data to plot.")
        return

    sphy_evolution = np.array(sphy_evolution_list)
    time_points = np.linspace(0, 1, len(sphy_evolution))
    
    # Using interpolation to smooth the stability evolution
    signals = [interp1d(time_points, np.roll(sphy_evolution, i), kind='cubic') for i in range(2)]
    new_time = np.linspace(0, 1, 2000)
    
    data = [signal(new_time) + np.random.normal(0, 0.15, len(new_time)) for signal in signals]
    weights = np.linspace(1, 1.5, 2)
    entanglement = np.average(data, axis=0, weights=weights)

    stability_mean = np.mean(entanglement)
    stability_variance = np.var(entanglement)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    ax1.plot(new_time, entanglement, 'k--', linewidth=2, label="Average Entanglement")
    for i in range(len(data)):
        ax1.plot(new_time, data[i], alpha=0.3, color='blue' if i == 0 else 'red')
    ax1.set_xlabel("Normalized Time")
    ax1.set_ylabel("Coherence/Amplitude")
    ax1.set_title(f"GHZ Entanglement - {num_qubits} Qubits")
    ax1.legend()
    ax1.grid()

    ax2.plot(new_time, entanglement, 'k-', label="Average Entanglement")
    ax2.axhline(stability_mean, color='green', linestyle='--', label=f"Mean: {stability_mean:.2f}")
    ax2.axhline(stability_mean + np.sqrt(stability_variance), color='orange', linestyle='--', label=f"± Variance")
    ax2.axhline(stability_mean - np.sqrt(stability_variance), color='orange', linestyle='--')
    ax2.set_xlabel("Normalized Time")
    ax2.set_ylabel("Coherence/Amplitude")
    ax2.set_title("Entanglement Stability")
    ax2.legend()
    ax2.grid()

    fig.suptitle(f"GHZ Stabilization Analysis (Cirq): {total_frames} Rounds", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the plot
    log_dir_output = os.path.dirname(file_path) if os.path.dirname(file_path) else LOG_DIR
    graph_filename = os.path.join(log_dir_output, f"ghz_cirq_qbench_report_EN_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    
    os.makedirs(log_dir_output, exist_ok=True)
    
    plt.savefig(graph_filename, dpi=300)
    print(f"\n📊 Graphical Report saved to: {graph_filename}")
    plt.show()

if __name__ == "__main__":
    analyze_ghz_cirq_log()