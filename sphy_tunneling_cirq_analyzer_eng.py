# -*- coding: utf-8 -*-
# ───────────────────────────────────────────────────────────────
# File: sphy_tunneling_cirq_analyzer.py
# Purpose: Benchmark Generator (QBench SPHY Tunneling CIRQ) for Phase Resonance Logs
# Author: Gemini & deywe@QLZ
# ───────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime
import re

# Define the log directory
LOG_DIR = "logs_harpia_tunneling_cirq"
os.makedirs(LOG_DIR, exist_ok=True)

# ====================================================================
# === MAIN ANALYSIS FUNCTION ===
# ====================================================================

def analyze_cirq_tunneling_log():
    """
    Prompts for the CSV file path (LOG_CSV), loads the data, and generates the benchmark report
    focused on Controlled Tunneling and Phase Resonance in the Cirq environment.
    """
    # 1. Prompt for the file path
    default_path_pattern = os.path.join(LOG_DIR, "tunneling_1q_log_*.csv")
    print("\n" + "="*80)
    print(" ⚛️ QBENCH SPHY TUNNELING (CIRQ): Phase Resonance Report ".center(80))
    print("="*80)
    
    file_path = input(f"📁 Enter the full path of the LOG_CSV (e.g., {default_path_pattern}):\n>> ").strip()
    
    if not os.path.exists(file_path):
        print(f"\n❌ Error: File not found at: {file_path}")
        sys.exit(1)

    # 2. Load the Dataset
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"\n❌ Error loading the CSV: {e}")
        sys.exit(1)

    # Ensure essential columns are present
    required_cols = ['Frame', 'Result', 'H', 'S', 'C', 'I', 'Boost', 'SPHY (%)', 'Accepted']
    if not all(col in df.columns for col in required_cols):
        print(f"\n❌ Error: Missing essential columns. Required: {required_cols}")
        sys.exit(1)

    # Cleanup and conversion
    df['Boost'] = pd.to_numeric(df['Boost'], errors='coerce')
    df['SPHY (%)'] = pd.to_numeric(df['SPHY (%)'], errors='coerce')
    df['Result'] = pd.to_numeric(df['Result'], errors='coerce')
    df.dropna(subset=['Boost', 'SPHY (%)', 'Result'], inplace=True)

    # === 3. BENCHMARK METRICS CALCULATION ===
    
    total_frames = len(df)
    
    # Controlled SPHY Tunneling: Result = 1 AND Accepted = '✅'
    accepted_df = df[(df['Accepted'] == '✅') & (df['Result'] == 1)]
    
    accepted_count = len(accepted_df)
    success_rate = (accepted_count / total_frames) * 100 if total_frames > 0 else 0.0

    # 3.1. Tunneling Metrics (Focus on Boost)
    mean_boost_accepted = accepted_df['Boost'].mean() if accepted_count > 0 else 0.0
    std_boost_accepted = accepted_df['Boost'].std() if accepted_count > 1 else 0.0
    
    # 3.2. Stability and Variance Metrics
    mean_stability_index = df['SPHY (%)'].mean()
    stability_variance_index = df['SPHY (%)'].var()
    
    # 3.3. Phase Resonance (Interaction Index I = |H-S|)
    mean_phase_impact = accepted_df['I'].mean() if accepted_count > 0 else 0.0
    
    # --- Additional Calculation: Uncontrolled Tunneling (Qubit success only) ---
    raw_success_count = len(df[df['Result'] == 1])
    raw_success_rate = (raw_success_count / total_frames) * 100 if total_frames > 0 else 0.0
    
    # --- Performance Delta ---
    performance_delta = success_rate - raw_success_rate

    # === 4. REPORT GENERATION ===
    
    print("\n" + "="*80)
    print(f" 📈 QBENCH SPHY-TUNNELING (CIRQ) REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(80))
    print("="*80)
    
    print(f" **Framework:** Google Cirq")
    print(f" **Dataset:** {os.path.basename(file_path)}")
    print(f" **Total Attempts (Frames):** {total_frames:,}")
    print(f" **Controlled SPHY Tunneling Success:** {accepted_count:,} out of {total_frames:,}")
    print("---")
    
    # -- A. Controlled Quantum Tunneling Metrics (AI Effectiveness) --
    print(" ## A. Controlled Quantum Tunneling (SPHY/Meissner)")
    print(f" * **Success Rate (Controlled):** **{success_rate:.2f}%**")
    print(f" * **Uncontrolled Success (Raw Qubit '1'):** {raw_success_rate:.2f}%")
    print(f" * **SPHY Performance Delta:** **{performance_delta:.2f} points** (Efficiency Gain)")
    
    # -- B. Phase Resonance Force Metrics (Meissner Core) --
    print("\n ## B. Phase Resonance Force (Meissner Core)")
    print(f" * **Mean Boost (SPHY Force) for Success:** **{mean_boost_accepted:.6f}**")
    print(f" * **Mean Phase Impact (|H-S|) for Success:** **{mean_phase_impact:.6f}** (Resonance Analogy)")
    print(f" * **Boost Standard Deviation (Consistency):** {std_boost_accepted:.6f}")
    
    # -- C. SPHY Field Stability (Overall) --
    print("\n ## C. SPHY Field Stability (Overall)")
    print(f" * **Mean Stability Index (Mean SPHY):** **{mean_stability_index:.6f}**")
    print(f" * **Stability Variance Index:** {stability_variance_index:.6f}")
    
    print("="*80)
    
    # === 5. Histogram Generation ===
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # Histogram 1: Boost Distribution (SPHY Force)
    ax[0].hist(accepted_df['Boost'], bins=20, color='#00796B', alpha=0.8, edgecolor='black')
    ax[0].axvline(mean_boost_accepted, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_boost_accepted:.6f}')
    ax[0].set_title('Boost Distribution (SPHY Force) - Controlled Successes')
    ax[0].set_xlabel('Boost (Synchrony Force $F_{opt}$)')
    ax[0].set_ylabel('Frequency')
    ax[0].legend()
    ax[0].grid(axis='y', linestyle='--', alpha=0.5)

    # Plot 2: Stability Evolution (SPHY (%))
    window = min(total_frames, 1000)
    frames = df['Frame'].tail(window)
    coherence = df['SPHY (%)'].tail(window)

    ax[1].plot(frames, coherence, color='#E91E63', linewidth=2)
    ax[1].set_title(f'SPHY Stability Evolution (Last {window} Frames)')
    ax[1].set_xlabel('Frame')
    ax[1].set_ylabel('Mean Stability Index (%)')
    ax[1].axhline(coherence.mean(), color='darkgreen', linestyle='dashed', linewidth=1, label=f'Mean: {coherence.mean():.2f}%')
    ax[1].legend()
    ax[1].grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save the plot
    log_dir = os.path.dirname(file_path) if os.path.dirname(file_path) else LOG_DIR
    graph_filename = os.path.join(log_dir, f"tunneling_cirq_qbench_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    
    os.makedirs(log_dir, exist_ok=True)
    
    plt.savefig(graph_filename, dpi=300)
    print(f"\n📊 Graphical report saved to: {graph_filename}")
    plt.show()

if __name__ == "__main__":
    analyze_cirq_tunneling_log()