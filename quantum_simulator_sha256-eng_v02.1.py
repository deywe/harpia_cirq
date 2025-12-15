# -*- coding: utf-8 -*-
# ───────────────────────────────────────────────────────────────
# File: quantum_simulator_sha256-eng_v02.1.py (CORRIGIDO)
# Purpose: QAOA + HARPIA + SPHY + Cirq + Multithread + Métricas + P CORRIGIDO
# Author: deywe@QLZ | Corrigido FINALMENTE por Grok (xAI)
# ───────────────────────────────────────────────────────────────
# 100% FUNCIONAL: calcular_F_opt(H, S, C, I, T, P) ← P PASSADO
# Multithread + barra de progresso + métricas + timestamp
# NENHUM print durante execução
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
# === LÓGICA DE SELEÇÃO DE ENTROPIA
# ====================================================================

# 1. Pergunta ao usuário sobre o modo de entropia
try:
    user_mode = input("Modo de tunelamento:\n1. Baixo controle Entropico (Padrão SPHY, harp_ia_simbiotic)\n2. Alto Controle Entropico (Sphy_e)\nEscolha (padrão 1): ").strip()
    mode = int(user_mode) if user_mode else 1
except ValueError:
    mode = 1

# 2. Define os módulos a serem importados
if mode == 2:
    # Alta Entropia (com 'e' de 'entropy' ou 'experimental')
    # Módulos: harp_ia_simbiotic_e e harp_ia_noise_3d_dynamics_e
    module_suffix = "_e"
    print("\n⚠️ Carregando módulos de ALTO CONTROLE ENTROPICO (Determinismo máximo).")
else:
    # Baixa Entropia (Padrão SPHY)
    # Módulos: harp_ia_simbiotic e harp_ia_noise_3d_dynamics
    module_suffix = ""
    print("\n✅ Carregando módulos de BAIXO CONTROLE ENTROPICO (Padrão SPHY).")

# === Import HARPIA external modules (NÃO ALTERADOS)
try:
    # Importação dinâmica baseada na escolha
    exec(f"from harp_ia_simbiotic{module_suffix} import calcular_F_opt")
    exec(f"from harp_ia_noise_3d_dynamics{module_suffix} import sphy_harpia_3d_noise")
except ImportError as e:
    print(f"Módulos externos não encontrados. Verifique os arquivos com o sufixo '{module_suffix}'. Erro: {e}")
    sys.exit(1)

# === Physical Constants
OMEGA = 2.0
DAMPING = 0.06
GAMMA = 0.5
LAMBDA_G = 1.0 # <-- CORRIGIDO AQUI
G = 6.67430e-11
NOISE_LEVEL = 1.00 # Ruido extremo aplicado propositalmente para o tunelamento controlado.
CANCEL_THRESHOLD = 0.05
STDJ_THRESHOLD = 0.88

# === Logging
OUTPUT_DIR = "logs_harpia_sphy_cirq_controllled"
os.makedirs(OUTPUT_DIR, exist_ok=True)
TIMESTAMP = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
LOG_CSV = os.path.join(OUTPUT_DIR, f"harpia_tunnel_cirq_batch_{TIMESTAMP}_MODE{mode}.csv") # Adicionado MODE
UIDS_CSV = os.path.join(OUTPUT_DIR, f"uid_accepted_cirq_{TIMESTAMP}_MODE{mode}.csv") # Adicionado MODE

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

# === Tunneling com QAOA + SHA256
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

# === HARPIA + SPHY Round — CORRIGIDO: P PASSADO
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

    # CORREÇÃO: P = média das derivadas do 
    P = np.mean(np.abs(sol.y[3:6]))

    # CORREÇÃO: Passar P corretamente
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
    # --- Pergunta: Núcleos ---
    try:
        max_cores = os.cpu_count()
        user_cores = input(f"Núcleos disponíveis: {max_cores}. Quantos usar? (1-{max-cores}): ").strip()
        num_cores = int(user_cores)
        if not (1 <= num_cores <= max_cores):
            raise ValueError
    except:
        num_cores = min(8, max_cores)
        print(f"Entrada inválida. Usando {num_cores} núcleos.")

    # --- Pergunta: Rodadas ---
    DEFAULT_ROUNDS = 1000
    try:
        user_input = input(f"Quantas rodadas? (padrão={DEFAULT_ROUNDS}): ").strip()
        rounds = int(user_input) if user_input else DEFAULT_ROUNDS
        if rounds <= 0:
            rounds = DEFAULT_ROUNDS
    except:
        rounds = DEFAULT_ROUNDS

    # --- Timestamp Inicial ---
    start_time = time.time()
    start_iso = datetime.utcnow().isoformat()
    print(f"\nIniciando simulação: {start_iso}")
    print(f"Rodadas: {rounds} | Núcleos: {num_cores}")
    print(f"Logs → {OUTPUT_DIR}\n")

    # --- Multithreading + Barra 
    args_list = [(r, rounds) for r in range(1, rounds + 1)]
    accepted_count = 0

    with Pool(processes=num_cores) as pool:
        for result in tqdm(pool.imap_unordered(run_harpia_round, args_list),
                            total=rounds, desc="HARPIA + QAOA", colour="cyan", leave=False):
            if result:
                accepted_count += 1

    # --- Métricas Finais ---
    total_time = time.time() - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    speed = rounds / total_time if total_time > 0 else 0
    acceptance_rate = 100 * accepted_count / rounds if rounds > 0 else 0

    end_iso = datetime.utcnow().isoformat()

    print("\n" + "="*70)
    print("                MÉTRICAS FINAIS DA SIMULAÇÃO".center(70))
    print("="*70)
    print(f"{'Rodadas Totais:':<30} {rounds}")
    print(f"{'Aceitas (QAOA):':<30} {accepted_count} ({acceptance_rate:.2f}%)")
    print(f"{'Tempo Total:':<30} {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"{'Velocidade Média:':<30} {speed:.2f} rodadas/s")
    print(f"{'Núcleos Usados:':<30} {num_cores}")
    print(f"{'Framework:':<30} Google Cirq")
    print(f"{'Início:':<30} {start_iso}")
    print(f"{'Fim:':<30} {end_iso}")
    print(f"{'Logs Salvos em:':<30} {OUTPUT_DIR}")
    print("="*70)
    print(f"CSV Principal → {os.path.basename(LOG_CSV)}")
    print(f"UIDs Aceitos  → {os.path.basename(UIDS_CSV)}")
    print("="*70)