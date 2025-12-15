# -*- coding: utf-8 -*-
# ───────────────────────────────────────────────────────────────
# File: sphy_cirq_qiskit_v2_satellite.py
# Purpose: GHZ + HARPIA (Cirq) + Adaptive Coherence Simulation + Meissner IA
# Author: deywe@QLZ | Rewritten by Gemini
# ───────────────────────────────────────────────────────────────

# 1️⃣ Importação dos módulos necessários: IA Meissner, estabiliza os qubits eticos usando gravidade sob o efeito meissiner
from meissner_core import meissner_correction_step
from sphy_satellite_module import simular_caminho_satelite, campo_de_harmonia_vibracional

import cirq
import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import datetime
import os, random, sys, time, hashlib, re
from tqdm import tqdm
from multiprocessing import Pool, Manager
from scipy.interpolate import interp1d

# Define the noise channel as a custom Cirq noise model
def custom_bit_flip_channel(prob):
    """A custom bit flip channel for simulating a given probability of error."""
    return cirq.X**2 * (1 - prob) + cirq.X * prob

# Directories and Parameters
LOG_DIR = "logs_harpia"
os.makedirs(LOG_DIR, exist_ok=True)

def get_user_parameters():
    try:
        num_qubits = int(input("🔢 Number of Qubits in GHZ circuit: "))
        total_pairs = int(input("🔁 Total GHZ states to simulate: "))
        return num_qubits, total_pairs
    except ValueError:
        print("❌ Invalid input. Please enter integers.")
        exit(1)

def generate_ghz_state_cirq(num_qubits, noise_prob=1.0):
    qubits = cirq.LineQubit.range(num_qubits)
    circuit = cirq.Circuit()
    
    circuit.append(cirq.H(qubits[0]))
    for i in range(1, num_qubits):
        circuit.append(cirq.CNOT(qubits[0], qubits[i]))

    if num_qubits > 1:
        qubit_to_noise = random.randint(1, num_qubits - 1)
        circuit.append(cirq.BitFlipChannel(p=noise_prob).on(qubits[qubit_to_noise]))

    circuit.append(cirq.measure(*qubits, key='result'))
    
    return circuit

def simulate_frame_cirq(frame_data):
    frame, num_qubits, total_frames, satellite_positions, sphy_coherence = frame_data
    random.seed(os.getpid() * frame)
    simulator = cirq.Simulator()
    ideal_states = ['0' * num_qubits, '1' * num_qubits]

    current_timestamp = datetime.utcnow().isoformat()
    
    posicao_satelite = satellite_positions[frame - 1]
    orbital_harmony = campo_de_harmonia_vibracional(posicao_satelite)
    
    noise_prob = max(0.0, min(1.0, 1.0 - orbital_harmony))
    
    circuit = generate_ghz_state_cirq(num_qubits, noise_prob)

    result_cirq = simulator.run(circuit, repetitions=1)
    raw_bits = result_cirq.measurements['result'][0]
    result = ''.join(map(str, np.array(raw_bits).astype(int)))

    H = random.uniform(0.95, 1.0)
    S = random.uniform(0.95, 1.0)
    C = sphy_coherence / 100
    I = abs(H - S)
    T = frame

    psi_state = [3.0, 3.0, 1.2, 1.2, 0.6, 0.5]

    try:
        boost, phase_impact, psi_state = meissner_correction_step(H, S, C, I, T, psi_state)
    except Exception as e:
        return None, None, f"\nCritical error in frame {frame} (Meissner IA): {e}"

    delta = boost * 0.7
    new_coherence = min(100, sphy_coherence + delta)
    activated = delta > 0
    accepted = (result in ideal_states) and activated

    data_to_hash = f"{frame}:{result}:{H:.4f}:{S:.4f}:{C:.4f}:{I:.4f}:{boost:.4f}:{new_coherence:.4f}:{current_timestamp}"
    sha256_signature = hashlib.sha256(data_to_hash.encode('utf-8')).hexdigest()

    log_entry = [
        frame, result, round(H, 4), round(S, 4),
        round(C, 4), round(I, 4), round(boost, 4),
        round(new_coherence, 4), "✅" if accepted else "❌",
        sha256_signature, current_timestamp
    ]
    
    # ❗ Retorna o impacto da correção de fase
    return log_entry, new_coherence, phase_impact, None

def execute_simulation_multiprocessing_cirq(num_qubits, total_frames=100000, num_processes=4):
    print("=" * 60)
    print(f" 🧿 HARPIA QGHZ STABILIZER + Meissner • {num_qubits} Qubits • {total_frames:,} Frames")
    print("=" * 60)

    timecode = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(LOG_DIR, f"qghz_{num_qubits}q_log_{timecode}.csv")
    fig_filename = os.path.join(LOG_DIR, f"qghz_{num_qubits}q_graph_{timecode}.png")

    manager = Manager()
    sphy_coherence = manager.Value('f', 90.0)
    log_data, sphy_evolution = manager.list(), manager.list()
    valid_states = manager.Value('i', 0)
    
    # ❗ Novos geradores para as métricas Phi
    total_phi_corrections = manager.Value('f', 0.0)
    individual_phi_corrections = manager.list()

    print("🛰️ Gerando caminho orbital do satélite...")
    satellite_positions = simular_caminho_satelite(total_frames)
    
    frame_inputs = [(f, num_qubits, total_frames, satellite_positions, sphy_coherence.value) for f in range(1, total_frames + 1)]

    print(f"🔄 Using {num_processes} processes for simulation...")
    with Pool(processes=num_processes) as pool:
        for result in tqdm(pool.imap_unordered(simulate_frame_cirq, frame_inputs),
                                                    total=total_frames, desc="⏳ Simulating GHZ"):
            log_entry, new_coherence, phase_impact, error = result
            if error:
                print(f"\n{error}", file=sys.stderr)
                pool.terminate()
                break
            if log_entry:
                log_data.append(log_entry)
                sphy_evolution.append(new_coherence)
                sphy_coherence.value = new_coherence
                if log_entry[-3] == "✅":
                    valid_states.value += 1
                
                # ❗ Acumula as correções de fase
                total_phi_corrections.value += abs(phase_impact)
                individual_phi_corrections.append(abs(phase_impact))

    acceptance_rate = 100 * (valid_states.value / total_frames) if total_frames > 0 else 0
    print(f"\n✅ GHZ States accepted: {valid_states.value}/{total_frames} | {acceptance_rate:.2f}%")
    
    if sphy_evolution:
        sphy_np_array = np.array(sphy_evolution)
        mean_stability = np.mean(sphy_np_array)
        stability_variance = np.var(sphy_np_array)
        print(f"\n📊 Mean Stability Index: {mean_stability:.6f}")
        print(f"📊 Stability Variance Index: {stability_variance:.6f}")

    # ❗ Exibe as novas métricas de correção orbital
    if individual_phi_corrections:
        mean_phi = np.mean(individual_phi_corrections)
        variance_phi = np.var(individual_phi_corrections)
        print("\n⚙️ SPHY Orbital Correction Metrics")
        print(f"    Total Gravitational Corrections (Φ): {total_phi_corrections.value:.4f}")
        print(f"    Mean Phase Correction (Φ): {mean_phi:.6f}")
        print(f"    Variance (Φ): {variance_phi:.6f}")

    with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Frame", "Result", "H", "S", "C", "I", "Boost", "SPHY (%)", "Accepted", "SHA256_Signature", "Timestamp"])
        writer.writerows(list(log_data))
    print(f"🧾 CSV saved: {csv_filename}")

    sphy_evolution_list = list(sphy_evolution)
    if not sphy_evolution_list:
        print("❌ No data to plot.")
        return

    sphy_evolution = np.array(sphy_evolution_list)
    tempo = np.linspace(0, 1, len(sphy_evolution))
    sinais = [interp1d(tempo, np.roll(sphy_evolution, i), kind='cubic') for i in range(2)]
    novo_tempo = np.linspace(0, 1, 2000)
    dados = [sinal(novo_tempo) + np.random.normal(0, 0.15, len(novo_tempo)) for sinal in sinais]
    pesos = np.linspace(1, 1.5, 2)
    emaranhamento = np.average(dados, axis=0, weights=pesos)
    estabilidade_media = np.mean(emaranhamento)
    estabilidade_variancia = np.var(emaranhamento)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    ax1.plot(novo_tempo, emaranhamento, 'k--', linewidth=2, label="Average Entanglement")
    for i in range(len(dados)):
        ax1.plot(novo_tempo, dados[i], alpha=0.3, color='blue' if i == 0 else 'red')
    ax1.set_xlabel("Normalized Time")
    ax1.set_ylabel("Coherence/Amplitude")
    ax1.set_title(f"GHZ Entanglement - {num_qubits} Qubits")
    ax1.legend()
    ax1.grid()

    ax2.plot(novo_tempo, emaranhamento, 'k-', label="Average Entanglement")
    ax2.axhline(estabilidade_media, color='green', linestyle='--', label=f"Mean: {estabilidade_media:.2f}")
    ax2.axhline(estabilidade_media + np.sqrt(estabilidade_variancia), color='orange', linestyle='--', label=f"± Variance")
    ax2.axhline(estabilidade_media - np.sqrt(estabilidade_variancia), color='orange', linestyle='--')
    ax2.set_xlabel("Normalized Time")
    ax2.set_ylabel("Coherence/Amplitude")
    ax2.set_title("Entanglement Stability")
    ax2.legend()
    ax2.grid()

    fig.suptitle(f"GHZ Simulation: Entanglement and Stability - {num_qubits} Qubits", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(fig_filename, dpi=300)
    plt.show()

if __name__ == "__main__":
    qubits, pairs = get_user_parameters()
    execute_simulation_multiprocessing_cirq(num_qubits=qubits, total_frames=pairs, num_processes=8)