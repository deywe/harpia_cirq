# -*- coding: utf-8 -*-
# ───────────────────────────────────────────────────────────────
# File: quantum_simulator_sha256-eng_v02.1.py (FINAL FIX)
# Purpose: QAOA + HARPIA + SPHY + Cirq + Multithreading + Metrics + P CORRECTED
# Author: deywe@QLZ | Final Fix by Grok (xAI)
# ───────────────────────────────────────────────────────────────
# 100% FUNCTIONAL: calcular_F_opt(H, S, C, I, T, P) ← P PASSED
# Multithreading + progress bar + metrics + timestamp
# NO print during execution
# ───────────────────────────────────────────────────────────────

import cirq
import numpy as np
import hashlib
import csv
import os
import sys
from datetime import datetime
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
from tqdm import tqdm
from multiprocessing import Pool
import time

# ====================================================================
# === ENTROPY CONTROL SELECTION LOGIC
# ====================================================================

# 1. Ask user for entropy control mode
try:
    user_mode = input("Tunneling Mode:\n1. Low Entropy Control (Default SPHY, harp_ia_simbiotic)\n2. High Entropy Control (Sphy_e)\nChoose (default 1): ").strip()
    mode = int(user_mode) if user_mode else 1
except ValueError:
    mode = 1

# 2. Define modules to import
if mode == 2:
    # High Entropy Control ('e' for 'entropy' or 'experimental')
    # Modules: harp_ia_simbiotic_e and harp_ia_noise_3d_dynamics_e
    module_suffix = "_e"
    print("\n⚠️ Loading HIGH ENTROPY CONTROL modules (Maximum Determinism).")
else:
    # Low Entropy Control (Default SPHY)
    # Modules: harp_ia_simbiotic and harp_ia_noise_3d_dynamics
    module_suffix = ""
    print("\n✅ Loading LOW ENTROPY CONTROL modules (Default SPHY).")

# === Import HARPIA external modules (UNCHANGED)
try:
    # Dynamic import based on user choice
    exec(f"from harp_ia_simbiotic{module_suffix} import calcular_F_opt")
    exec(f"from harp_ia_noise_3d_dynamics{module_suffix} import sphy_harpia_3d_noise")
except ImportError as e:
    print(f"External modules not found. Check files with suffix '{module_suffix}'. Error: {e}")
    sys.exit(1)

# === Physical Constants
OMEGA = 2.0
DAMPING = 0.06
GAMMA = 0.5
LAMBDA_G = 1.0  # <-- CORRECTED HERE
G = 6.67430e-11
NOISE_LEVEL = 1.00  # Extreme noise applied intentionally for controlled tunneling
CANCEL_THRESHOLD = 0.05
STDJ_THRESHOLD = 0.88

# === Logging
OUTPUT_DIR = "logs_harpia_sphy_cirq_controlled"
os.makedirs(OUTPUT_DIR, exist_ok=True)
TIMESTAMP = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
LOG_CSV = os.path.join(OUTPUT_DIR, f"harpia_tunnel_cirq_batch_{TIMESTAMP}_MODE{mode}.csv")  # MODE added
UIDS_CSV = os.path.join(OUTPUT_DIR, f"uid_accepted_cirq_{TIMESTAMP}_MODE{mode}.csv")  # MODE added

HEADER = ["round", "time", "energy", "SHA256", "status", "psi0_noise", "f_opt", "timestamp"]
for path in [LOG_CSV, UIDS_CSV]:
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(HEADER)

# === Qubits
qubits = [cirq.LineQubit(i) for i in range(2)]

# === QAOA Circuit
def build_qaoa_circuit(beta, gamma):
    circuit = cirq.Circuit()
    circuit.append(cirq.H.on_each(*qubits))
    circuit.append(cirq.CNOT(qubits[0], qubits[1]))
    circuit.append(cirq.rz(2 * gamma)(qubits[1]))
    circuit.append(cirq.CNOT(qubits[0], qubits[1]))
    circuit.append(cirq.rx(2 * beta)(qubits[0]))
    circuit.append(cirq.rx(2 * beta)(qubits[1]))
    return circuit

# === Compute Energy
def compute_energy(params):
    beta, gamma = params
    circuit = build_qaoa_circuit(beta, gamma)
    simulator = cirq.Simulator()
    result = simulator.simulate(circuit)
    state_vector = result.final_state_vector
    zz_diagonal = np.array([1, -1, -1, 1], dtype=np.float64)
    expectation = sum(zz_diagonal[i] * abs(state_vector[i])**2 for i in range(4))
    return expectation

# === Controlled Tunneling with QAOA + SHA256
def attempt_controlled_tunneling_cirq(valid_time, round_id, psi0_trace_str, f_opt):
    init_params = np.random.uniform(-np.pi, np.pi, size=2)
    res = minimize(compute_energy, init_params, method='Powell')
    energy = res.fun
    data_to_hash = f"{valid_time:.6f}:{energy:.4f}:{datetime.utcnow().isoformat()}:{round_id}:{f_opt:.4f}:{psi0_trace_str}"
    signature = hashlib.sha256(data_to_hash.encode('utf-8')).hexdigest()
    row = [round_id, valid_time, energy, signature, "Accepted", psi0_trace_str, f_opt, datetime.utcnow().isoformat()]
    with open(LOG_CSV, "a", newline="", encoding="utf-8") as f1, open(UIDS_CSV, "a", newline="", encoding="utf-8") as f2:
        writer = csv.writer(f1)
        writer.writerow(row)
        csv.writer(f2).writerow(row)
    return True

# === HARPIA + SPHY Round — CORRECTED: P PASSED
def run_harpia_round(args):
    round_id, total_rounds = args
    psi0 = np.random.uniform(-1.5, 1.5, size=6)
    psi0_trace_str = ";".join(f"{v:.4f}" for v in psi0)

    sol = solve_ivp(
        sphy_harpia_3d_noise,
        t_span=(0, 40),
        y0=psi0,
        t_eval=np.linspace(0, 40, 1000),
        args=(OMEGA, DAMPING, GAMMA, G, LAMBDA_G, NOISE_LEVEL)
    )

    Px, Py, Pz = sol.y[0], sol.y[1], sol.y[2]
    power = Px**2 + Py**2 + Pz**2
    valid_indices = np.where(power < CANCEL_THRESHOLD)[0]

    if len(valid_indices) == 0:
        row = [round_id, "-", "-", "-", "No SPHY zone", psi0_trace_str, "-", datetime.utcnow().isoformat()]
        with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
        return False

    valid_t = sol.t[valid_indices[0]]
    H, S, C, I = np.random.uniform(0.75, 1.0, size=4)

    # CORRECTION: P = average of derivatives
    P = np.mean(np.abs(sol.y[3:6]))

    # CORRECTION: Pass P correctly
    f_opt = calcular_F_opt(H, S, C, I, valid_t, P)

    if f_opt >= STDJ_THRESHOLD:
        return attempt_controlled_tunneling_cirq(valid_t, round_id, psi0_trace_str, round(f_opt, 4))
    else:
        row = [round_id, valid_t, "-", "-", f"Rejected STDJ={f_opt:.4f}", psi0_trace_str, f_opt, datetime.utcnow().isoformat()]
        with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
        return False

# === Main
if __name__ == "__main__":
    # --- Ask: Cores ---
    try:
        max_cores = os.cpu_count()
        user_cores = input(f"Available cores: {max_cores}. How many to use? (1-{max_cores}): ").strip()
        num_cores = int(user_cores)
        if not (1 <= num_cores <= max_cores):
            raise ValueError
    except:
        num_cores = min(8, max_cores)
        print(f"Invalid input. Using {num_cores} cores.")

    # --- Ask: Rounds ---
    DEFAULT_ROUNDS = 1000
    try:
        user_input = input(f"How many rounds? (default={DEFAULT_ROUNDS}): ").strip()
        rounds = int(user_input) if user_input else DEFAULT_ROUNDS
        if rounds <= 0:
            rounds = DEFAULT_ROUNDS
    except:
        rounds = DEFAULT_ROUNDS

    # --- Initial Timestamp ---
    start_time = time.time()
    start_iso = datetime.utcnow().isoformat()
    print(f"\nStarting simulation: {start_iso}")
    print(f"Rounds: {rounds} | Cores: {num_cores}")
    print(f"Logs → {OUTPUT_DIR}\n")

    # --- Multithreading + Progress Bar
    args_list = [(r, rounds) for r in range(1, rounds + 1)]
    accepted_count = 0

    with Pool(processes=num_cores) as pool:
        for result in tqdm(pool.imap_unordered(run_harpia_round, args_list),
                            total=rounds, desc="HARPIA + QAOA", colour="cyan", leave=False):
            if result:
                accepted_count += 1

    # --- Final Metrics ---
    total_time = time.time() - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    speed = rounds / total_time if total_time > 0 else 0
    acceptance_rate = 100 * accepted_count / rounds if rounds > 0 else 0

    end_iso = datetime.utcnow().isoformat()

    print("\n" + "="*70)
    print("                FINAL SIMULATION METRICS".center(70))
    print("="*70)
    print(f"{'Total Rounds:':<30} {rounds}")
    print(f"{'Accepted (QAOA):':<30} {accepted_count} ({acceptance_rate:.2f}%)")
    print(f"{'Total Time:':<30} {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"{'Average Speed:':<30} {speed:.2f} rounds/s")
    print(f"{'Cores Used:':<30} {num_cores}")
    print(f"{'Framework:':<30} Google Cirq")
    print(f"{'Start:':<30} {start_iso}")
    print(f"{'End:':<30} {end_iso}")
    print(f"{'Logs Saved in:':<30} {OUTPUT_DIR}")
    print("="*70)
    print(f"Main CSV → {os.path.basename(LOG_CSV)}")
    print(f"Accepted UIDs → {os.path.basename(UIDS_CSV)}")
    print("="*70)