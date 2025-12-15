# -*- coding: utf-8 -*-
# ───────────────────────────────────────────────────────────────
# File: cirq_sphy_dynamic_v4_tunnel_ENG.py
# Purpose: QUANTUM TUNNELING + HARPIA (Google Cirq) + SPHY Phase Resonance
# Author: deywe@QLZ | Adapted by Julliet AI & Gemini e Spock AI (Cirq Conversion)
# ───────────────────────────────────────────────────────────────

# 1️⃣ Import necessary modules
# ASSUMPTION: 'meissner_core.py' is available in the current directory
# The meissner core conducts quantum tunneling using the 3D SPHY gravitational field geometry.
from meissner_core import meissner_correction_step 

# ⚛️ Cirq Imports
import cirq
from cirq.sim.simulator import SimulationTrialResult

import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import datetime
import os, random, sys, time, hashlib, re
from tqdm import tqdm
from multiprocessing import Pool, Manager
from scipy.interpolate import interp1d

# === Global Qubit Variable
# We use a standard qubit (q0) for the tunneling analogy
QUBIT = cirq.GridQubit(0, 0) 

# === Log Directory
LOG_DIR = "logs_harpia_tunneling_cirq"
os.makedirs(LOG_DIR, exist_ok=True)

# === Configuration and Circuit Functions

def get_user_parameters():
    try:
        # For the purpose of the tunneling analogy, we use a single qubit for the particle state.
        num_qubits = 1 
        print(f"🔢 Number of Qubits for forced Tunneling analogy set to: {num_qubits}")
        total_pairs = int(input("🔁 Total Tunneling Attempts (Frames) to simulate: "))
        
        # Barrier Strength: mapped to the RZ rotation (0.0=No Barrier, 1.0=Maximum Barrier)
        barrier_strength_input = float(input("🚧 Barrier Strength (0.0 to 1.0, where 1.0 is the strongest barrier): "))
        if not (0.0 <= barrier_strength_input <= 1.0):
             print("❌ Barrier Strength must be between 0.0 and 1.0.")
             exit(1)
        # Maps the strength from 0.0 to 1.0 to a rotation from 0 to pi/2
        barrier_strength_theta = barrier_strength_input * np.pi / 2 
        
        return num_qubits, total_pairs, barrier_strength_theta
    except ValueError:
        print("❌ Invalid input. Please enter integers/floats.")
        exit(1)

def generate_tunneling_circuit(qubit: cirq.Qid, barrier_strength_theta: float, noise_prob: float = 1.00) -> cirq.Circuit:
    """
    Creates the quantum tunneling circuit in Cirq format.
    """
    circuit = cirq.Circuit()

    # 1. Initial State (Superposition for "tunneling attempt")
    circuit.append(cirq.H(qubit))

    # 2. Barrier (Potential Analogy: RZ affects the phase, crucial for SPHY)
    # Rz in Cirq accepts radians directly
    circuit.append(cirq.Rz(rads=barrier_strength_theta)(qubit))
    
    # 3. SPHY Noise (Simulation of external phase perturbation)
    if random.random() < noise_prob:
        # Applies a random phase perturbation (using Rz as Phase Gate, functionally equivalent)
        random_phase_angle = random.uniform(-np.pi/8, np.pi/8)
        circuit.append(cirq.Rz(rads=random_phase_angle)(qubit))
    
    # 4. Measurement (Success is analogous to measuring the state |1>, i.e., the particle "passed")
    circuit.append(cirq.measure(qubit, key='m'))
    
    return circuit

# === Main Simulation Function per Frame

def simulate_frame(frame_data):
    frame, num_qubits, total_frames, noise_prob, sphy_coherence, barrier_theta = frame_data
    # Use the globally defined QUBIT
    qubit = QUBIT 
    random.seed(os.getpid() * frame) 
    # 🔄 Cirq Simulator
    simulator = cirq.Simulator()
    # The ideal state for tunneling is to measure '1' (passed the barrier)
    ideal_state = 1 

    current_timestamp = datetime.utcnow().isoformat()
    
    # Tunneling Circuit Generation (Cirq)
    circuit = generate_tunneling_circuit(qubit, barrier_theta, noise_prob)
    
    # 🔄 Executes the simulation (1 shot)
    job_result: SimulationTrialResult = simulator.run(circuit, repetitions=1)
    
    # 🔄 Cirq Result Processing
    # The 'm' measurement result is a numpy array; we take the first (and only) shot
    result_raw = job_result.data['m'][0]
    result = int(result_raw)

    # === SPHY/Meissner Logic for Adaptive Coherence ===
    H = random.uniform(0.95, 1.0) # SPHY Harmonic (Gravitational)
    S = random.uniform(0.95, 1.0) # SPHY Synchrony
    C = sphy_coherence / 100    # Current Coherence (as SPHY input)
    I = abs(H - S)              # Interaction Index (Field Variation)
    T = frame                   # Time/Frame

    # Initial AI state (assuming it maintains the dynamic state)
    psi_state = [3.0, 3.0, 1.2, 1.2, 0.6, 0.5]

    try:
        # The Meissner/SPHY correction decides the best moment for tunneling
        boost, phase_impact, psi_state = meissner_correction_step(H, S, C, I, T, psi_state)
    except Exception as e:
        return None, None, f"\nCritical Error in frame {frame} (Meissner AI): {e}"

    delta = boost * 0.7
    new_coherence = min(100, sphy_coherence + delta)
    activated = delta > 0 # SPHY/Meissner activated the synchrony pulse
    
    # Tunneling is accepted if the final state is '1' (Passed) AND SPHY activated the pulse at the right moment
    accepted = (result == ideal_state) and activated

    data_to_hash = f"{frame}:{result}:{H:.4f}:{S:.4f}:{C:.4f}:{I:.4f}:{boost:.4f}:{new_coherence:.4f}:{current_timestamp}"
    sha256_signature = hashlib.sha256(data_to_hash.encode('utf-8')).hexdigest()

    log_entry = [
        frame, result, round(H, 4), round(S, 4),
        round(C, 4), round(I, 4), round(boost, 4),
        round(new_coherence, 4), "✅" if accepted else "❌",
        sha256_signature, current_timestamp
    ]
    return log_entry, new_coherence, None

# === Main Execution Function (Multiprocessing)

def execute_simulation_multiprocessing(num_qubits, total_frames, barrier_theta, noise_prob=0.3, num_processes=4):
    print("=" * 60)
    print(f" ⚛️ HARPIA-SPHY: Controlled Quantum Tunneling (CIRQ) • {total_frames:,} Frames")
    print(f" 🚧 Barrier Strength: {barrier_theta*180/np.pi:.2f} degrees RZ (Analog)")
    print("=" * 60)

    timecode = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(LOG_DIR, f"tunneling_{num_qubits}q_log_{timecode}.csv")
    fig_filename = os.path.join(LOG_DIR, f"tunneling_{num_qubits}q_graph_{timecode}.png")

    manager = Manager()
    sphy_coherence = manager.Value('f', 90.0)
    log_data, sphy_evolution = manager.list(), manager.list()
    valid_states = manager.Value('i', 0)
    
    # Pass the initial (and static, due to the Pool) coherence value for the frames
    # Note: Dynamic coherence control in multiprocessing with Pool is limited,
    # maintaining the original structure of your code.
    frame_inputs = [
        (f, num_qubits, total_frames, noise_prob, sphy_coherence.value, barrier_theta) 
        for f in range(1, total_frames + 1)
    ]

    print(f"🔄 Using {num_processes} processes for simulation...")
    with Pool(processes=num_processes) as pool:
        for log_entry, new_coherence, error in tqdm(pool.imap_unordered(simulate_frame, frame_inputs),
                                                    total=total_frames, desc="⏳ Simulating Tunneling"):
            if error:
                print(f"\n{error}", file=sys.stderr)
                pool.terminate()
                break
            if log_entry:
                log_data.append(log_entry)
                sphy_evolution.append(new_coherence)
                # Update the coherence value for the next round in the main process
                sphy_coherence.value = new_coherence 
                if log_entry[-3] == "✅":
                    valid_states.value += 1

    # --- Metrics Calculation and Printing ---
    acceptance_rate = 100 * (valid_states.value / total_frames) if total_frames > 0 else 0
    print(f"\n✅ Tunneling Success Rate (SPHY Controlled): {valid_states.value}/{total_frames} | {acceptance_rate:.2f}%")
    
    if sphy_evolution:
        sphy_np_array = np.array(sphy_evolution)
        mean_stability = np.mean(sphy_np_array)
        stability_variance = np.var(sphy_np_array)
        print(f"\n📊 Mean Stability Index (SPHY): {mean_stability:.6f}")
        print(f"📊 Stability Variance Index: {stability_variance:.6f}")

    with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Frame", "Result", "H", "S", "C", "I", "Boost", "SPHY (%)", "Accepted", "SHA256_Signature", "Timestamp"])
        writer.writerows(list(log_data))
    print(f"🧾 CSV saved: {csv_filename}")

    # --- Graph Generation (Maintained the GHZ/Entanglement format) ---
    sphy_evolution_list = list(sphy_evolution)
    if not sphy_evolution_list:
        print("❌ No data to plot.")
        return

    sphy_evolution = np.array(sphy_evolution_list)
    time_points = np.linspace(0, 1, len(sphy_evolution))
    
    # Using interpolation to smooth the tunneling stability evolution
    signals = [interp1d(time_points, np.roll(sphy_evolution, i), kind='cubic') for i in range(2)]
    new_time = np.linspace(0, 1, 2000)
    
    data = [sinal(new_time) + np.random.normal(0, 0.15, len(new_time)) for sinal in signals]
    weights = np.linspace(1, 1.5, 2)
    tunneling_stability = np.average(data, axis=0, weights=weights)

    stability_mean = np.mean(tunneling_stability)
    stability_variance = np.var(tunneling_stability)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    ax1.plot(new_time, tunneling_stability, 'k--', linewidth=2, label="SPHY Stability Evolution")
    for i in range(len(data)):
        ax1.plot(new_time, data[i], alpha=0.3, color='blue' if i == 0 else 'red')
    ax1.set_xlabel("Normalized Time")
    ax1.set_ylabel("Phase Stability/Amplitude")
    ax1.set_title(f"Controlled Quantum Tunneling - SPHY/Meissner (CIRQ)")
    ax1.legend()
    ax1.grid()

    ax2.plot(new_time, tunneling_stability, 'k-', label="Mean SPHY Stability")
    ax2.axhline(stability_mean, color='green', linestyle='--', label=f"Mean: {stability_mean:.2f}")
    ax2.axhline(stability_mean + np.sqrt(stability_variance), color='orange', linestyle='--', label=f"± Variance")
    ax2.axhline(stability_mean - np.sqrt(stability_variance), color='orange', linestyle='--')
    ax2.set_xlabel("Normalized Time")
    ax2.set_ylabel("Phase Stability/Amplitude")
    ax2.set_title("Tunneling Phase Stability")
    ax2.legend()
    ax2.grid()

    fig.suptitle(f"Quantum Tunneling Simulation: {total_frames} Attempts (CIRQ)", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.savefig(fig_filename, dpi=300)

if __name__ == "__main__":
    qubits, pairs, barrier_theta = get_user_parameters()
    # The number of processes was forced to 4 in the original, maintaining the structure.
    execute_simulation_multiprocessing(num_qubits=qubits, total_frames=pairs, barrier_theta=barrier_theta, noise_prob=0.3, num_processes=4)