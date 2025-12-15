# -*- coding: utf-8 -*-
# ───────────────────────────────────────────────────────────────
# File: sphy_simbiotic_qbench_analyzer_en.py
# Purpose: SPHY Simbiotic QBench Analyzer for GHZ Stabilization Logs (English)
# Author: Gemini & deywe@QLZ
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

# Define the log directory
LOG_DIR = "logs_harpia"
os.makedirs(LOG_DIR, exist_ok=True)

# ====================================================================
# === CORE ANALYSIS FUNCTION ===
# ====================================================================

def analyze_simbiotic_log():
    """
    Prompts for the CSV file path, loads the data, and generates the benchmark report
    focused on GHZ stability controlled by the Symbiotic SPHY core (English).
    """
    # 1. Prompt for file path
    default_path_pattern = os.path.join(LOG_DIR, "qghz_*q_log_*.csv")
    print("\n" + "="*80)
    print(" 📊 SPHY SIMBIOTIC QBENCH ANALYZER: GHZ Stability Report ".center(80))
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
    required_cols = ['Frame', 'Accepted', 'C', 'Boost', 'SPHY (%)', 'I']
    if not all(col in df.columns for col in required_cols):
        print(f"\n❌ Error: Essential columns missing. Required: {required_cols}")
        sys.exit(1)

    # Clean and convert columns
    df['Boost'] = pd.to_numeric(df['Boost'], errors='coerce')
    df['SPHY (%)'] = pd.to_numeric(df['SPHY (%)'], errors='coerce')
    df['C'] = pd.to_numeric(df['C'], errors='coerce')
    df.dropna(subset=['Boost', 'SPHY (%)', 'C'], inplace=True)

    # === 3. BENCHMARK METRICS CALCULATION ===
    
    total_frames = len(df)
    
    # Filter 'Accepted' states (Ideal GHZ state AND boost activated)
    accepted_df = df[df['Accepted'] == '✅']
    
    accepted_count = len(accepted_df)
    success_rate = (accepted_count / total_frames) * 100 if total_frames > 0 else 0.0

    # 3.1. Stability Metrics (Mean Stability Index / Variance)
    
    # Baseline Coherence (C) and Final Coherence (SPHY (%))
    baseline_coherence_mean = df['C'].mean() * 100 
    mean_stability_index = df['SPHY (%)'].mean()
    stability_variance_index = df['SPHY (%)'].var()
    
    # Net Gain: (Mean SPHY - Mean Baseline Coherence)
    coherence_net_gain = mean_stability_index - baseline_coherence_mean
    
    # 3.2. Control Metrics (Boost)
    mean_boost_accepted = accepted_df['Boost'].mean() if accepted_count > 0 else 0.0
    std_boost_accepted = accepted_df['Boost'].std() if accepted_count > 1 else 0.0
    
    # 3.3. Interaction (I)
    mean_interaction_index = df['I'].mean()
    
    # === 4. REPORT GENERATION (English) ===
    
    print("\n" + "="*80)
    print(f" 📈 SPHY SIMBIOTIC QBENCH REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(80))
    print("="*80)
    
    print(f" **Dataset:** {os.path.basename(file_path)}")
    print(f" **Total Rounds (Frames):** {total_frames:,}")
    print("---")
    
    # -- A. Symbiotic Filter Success Metrics --
    print(" ## A. Success Control (HARPIA Symbiotic Filter)")
    print(f" * **Final Acceptance Rate (GHZ + SPHY):** **{success_rate:.2f}%**")
    
    # -- B. Stability and Coherence Metrics (Core) --
    print("\n ## B. SPHY Stability Performance")
    print(f" * **Coherence Baseline Mean (C):** {baseline_coherence_mean:.4f}%")
    print(f" * **Mean Stability Index (SPHY Final):** **{mean_stability_index:.4f}%**")
    print(f" * **Net Stability Gain:** **{coherence_net_gain:.4f} points**")
    print(f" * **Stability Variance Index:** {stability_variance_index:.8f} (Stabilization Consistency)")
    
    # -- C. Symbiotic Field Action Metrics (Boost) --
    print("\n ## C. Symbiotic Synchrony Field Action (Boost)")
    print(f" * **Mean Boost (Accepted):** **{mean_boost_accepted:.6f}** (Applied Synchrony Force)")
    print(f" * **Boost Standard Deviation (Consistency):** {std_boost_accepted:.6f}")
    print(f" * **Mean Interaction Index (I=|H-S|):** {mean_interaction_index:.6f}")
    
    print("="*80)
    
    # === 5. Plot Generation (English) ===
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # Histogram 1: Boost Distribution
    ax[0].hist(accepted_df['Boost'], bins=30, color='darkcyan', alpha=0.8, edgecolor='black')
    ax[0].axvline(mean_boost_accepted, color='red', linestyle='dashed', linewidth=1, label=f'Accepted Mean: {mean_boost_accepted:.6f}')
    ax[0].set_title('Boost Distribution (Synchrony Force) - Accepted States ✅')
    ax[0].set_xlabel('Boost Value ($F_{opt}$)')
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
    graph_filename = os.path.join(log_dir, f"simbiotic_qbench_report_EN_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    
    os.makedirs(log_dir, exist_ok=True)
    
    plt.savefig(graph_filename, dpi=300)
    print(f"\n📊 Graphical Report saved to: {graph_filename}")
    plt.show()

if __name__ == "__main__":
    analyze_simbiotic_log()