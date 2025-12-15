import os
import csv
import sys
import re
import random
import numpy as np
import hashlib
import matplotlib.pyplot as plt
import subprocess
from tqdm import tqdm
from datetime import datetime
import cirq
import warnings
import matplotlib

# Suppress glyph-related warnings as a fallback
warnings.filterwarnings("ignore", category=UserWarning, message="Glyph .* missing from font")

# Try to set Noto Sans for Unicode support, fall back to sans-serif
try:
    plt.rcParams['font.family'] = 'Noto Sans'
except:
    plt.rcParams['font.family'] = 'sans-serif'

# Diretório para logs
LOG_DIR = "memory/entangled_states"
os.makedirs(LOG_DIR, exist_ok=True)

# Solicitando parâmetros do usuário (quantidade de qubits e pares)
def entrada_parametros():
    try:
        num_qubits = int(input("Number of Qubits in GHZ circuit (Cirq): "))
        total_pares = int(input("Total GHZ states to simulate: "))
        return num_qubits, total_pares
    except ValueError:
        print("Invalid input.")
        sys.exit(1)

def gerar_ghz_circuit_com_ruido(num_qubits: int, noise_prob: float = None):
    qubits = cirq.LineQubit.range(num_qubits)
    circuit = cirq.Circuit()

    # Solicita o nível de ruído se não fornecido
    if noise_prob is None:
        try:
            noise_prob = float(input("Enter depolarizing noise level (0.0 to 1.0): "))
            noise_prob = max(0.0, min(noise_prob, 1.0))  # Garante que está no intervalo [0, 1]
        except ValueError:
            print("Invalid value. Using noise_prob=0.01 as default.")
            noise_prob = 0.0  # Fixed default value

    circuit.append(cirq.H(qubits[0]))
    for i in range(1, num_qubits):
        circuit.append(cirq.CNOT(qubits[0], qubits[i]))

    # Adiciona ruído depolarizante em cada qubit
    for q in qubits:
        circuit.append(cirq.depolarize(p=noise_prob).on(q))

    circuit.append(cirq.measure(*qubits, key='m'))
    return circuit, qubits

# Função de medição com 1 shot
def medir(circuit, qubits):
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=1)
    bits = result.measurements["m"][0]
    return ''.join(str(b) for b in bits)

# Executável externo (mantido como no original)
def calcular_F_opt(H, S, C, I, T):
    try:
        result = subprocess.run(
            ["./sphy_simbiotic_entangle_ai", str(H), str(S), str(C), str(I), str(T)],
            capture_output=True, text=True, check=True
        )
        match = re.search(r"([-+]?\d*\.\d+|\d+)", result.stdout)
        return float(match.group(0)) if match else 0.0
    except:
        return 0.0

def gerar_uid_via_bscore():
    try:
        result = subprocess.run(
            ["./ai_validator_bscore_uid"],
            capture_output=True, text=True, check=True
        )
        linhas = result.stdout.strip().splitlines()
        for linha in linhas:
            if "UID aceita" in linha or "UID rejeitada" in linha:
                partes = linha.split("|")
                uid_info = partes[0].split(":")[1].strip()
                bscore_info = float(partes[1].replace("B(t) =", "").strip())
                status = "Aceita" if "UID aceita" in linha else "Rejeitada"
                return uid_info, bscore_info, status
        return "-", 0.0, "Erro"
    except Exception as e:
        print(f"Error running UID Rust: {e}")
        return "-", 0.0, "Erro"

# Loop principal
def executar_simulacao(num_qubits, total=100, noise_prob=0.01):
    # Timestamp for output
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")
    print("=" * 60)
    print(f"    HARPIA QGHZ (CIRQ) - {num_qubits} Qubits - {total:,} Frames")
    print(f"    Timestamp: {timestamp}")
    print("=" * 60)

    timecode = datetime.now().strftime("%Y%m%d_%H%M%S")
    nome_csv = os.path.join(LOG_DIR, f"qghz_cirq_{num_qubits}q_log_{timecode}.csv")
    nome_fig = os.path.join(LOG_DIR, f"qghz_cirq_{num_qubits}q_graph_{timecode}.png")

    sphy_coherence = 90.0
    validos = 0
    log_data = []
    sphy_evolution = []

    for frame in tqdm(range(1, total + 1), desc="Simulating GHZ Cirq"):
        circuit, qubits = gerar_ghz_circuit_com_ruido(num_qubits, noise_prob)
        resultado = medir(circuit, qubits)

        H = random.uniform(0.95, 1.0)
        S = random.uniform(0.95, 1.0)
        C = sphy_coherence / 100
        I = abs(H - S)
        T = frame

        boost = calcular_F_opt(H, S, C, I, T)
        delta = boost * 0.7
        novo = min(100, sphy_coherence + delta)

        uid_val, bscore_val, status_uid = gerar_uid_via_bscore()
        aceito = bscore_val >= 0.900
        if aceito:
            validos += 1

        sphy_coherence = novo
        sphy_evolution.append(sphy_coherence)

        log_line = [
            frame, resultado,
            round(H, 4), round(S, 4),
            round(C, 4), round(I, 4),
            round(boost, 4), round(sphy_coherence, 4),
            "Yes" if aceito else "No",
            uid_val, round(bscore_val, 4), status_uid
        ]
        uid_sha256 = hashlib.sha256(",".join(map(str, log_line)).encode()).hexdigest()
        log_line.append(uid_sha256)
        log_data.append(log_line)

        print(f"Frame {frame}: Res={resultado} | UID={uid_val} | B(t)={bscore_val:.4f} | Status={status_uid} | Accepted={'Yes' if aceito else 'No'}")
        sys.stdout.flush()

    acceptance_rate = 100 * (validos / total)
    mean_stability = np.mean(sphy_evolution)
    variance_stability = np.var(sphy_evolution)

    print(f"\nGHZ States accepted: {validos}/{total} | {acceptance_rate:.2f}%")
    print(f"Mean Stability Index: {mean_stability:.6f}")
    print(f"Stability Variance Index: {variance_stability:.6f}")
    print(f"Timestamp: {timestamp}")

    with open(nome_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Frame", "Result", "H", "S", "C", "I",
            "Boost", "SPHY (%)", "Accepted",
            "UID", "B(t)", "UID_Status", "UID_SHA256"
        ])
        writer.writerows(log_data)

    print(f"CSV saved: {nome_csv}")

    # Gráfico SPHY
    plt.figure(figsize=(12, 5))
    plt.scatter(
        range(1, total + 1), sphy_evolution,
        c=['green' if row[8] == "Yes" else 'red' for row in log_data],
        s=8, alpha=0.6,
        label="SPHY Coherence"
    )
    plt.axhline(90, color='gray', linestyle="dotted", linewidth=1, label="Threshold")
    plt.title(f"SPHY Evolution - {num_qubits} Qubits - {total:,} Frames (CIRQ)")
    plt.xlabel("Frames")
    plt.ylabel("SPHY Coherence (%)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(nome_fig, dpi=300)
    print(f"Graph saved: {nome_fig}")
    plt.show()

if __name__ == "__main__":
    qubits, pares = entrada_parametros()
    executar_simulacao(num_qubits=qubits, total=pares, noise_prob=0.01)