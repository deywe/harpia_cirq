# -*- coding: utf-8 -*-
# ───────────────────────────────────────────────────────────────
# File: sphy_simbiotic_v1.1.2_PT-Br_sha256.py (Modificado)
# Purpose: Simulacao GHZ + HARPIA (CIRQ) + Coerencia Adaptativa
# Author: deywe@QLZ | Adaptado para multithread por Gemini
# ───────────────────────────────────────────────────────────────

import cirq
import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import datetime
import os
import random
import subprocess
import re
from tqdm import tqdm
import sys
import hashlib
from multiprocessing import Pool, Manager
from multiprocessing import current_process

# 🔧 Configura pasta de logs
LOG_DIR = "logs_harpia"
os.makedirs(LOG_DIR, exist_ok=True)

# 🧠 Coletar parametros pelo usuario
def entrada_parametros():
    try:
        num_qubits = int(input("🔢 Numero de Qubits no circuito GHZ: "))
        total_pares = int(input("🔁 Total de estados GHZ a simular: "))
        return num_qubits, total_pares
    except ValueError:
        print("❌ Entrada invalida. Por favor, insira numeros inteiros.")
        exit(1)

# 🧬 Gerador GHZ com ruido simbolico
def gerar_ghz_state(nq, noise_prob=0.0):
    qbs = cirq.LineQubit.range(nq)
    circuit = cirq.Circuit()
    circuit.append(cirq.H(qbs[0]))
    for i in range(1, nq):
        circuit.append(cirq.CNOT(qbs[0], qbs[i]))
    if random.random() < noise_prob:
        circuit.append(cirq.X(random.choice(qbs[1:])))
    circuit.append(cirq.measure(*qbs, key='m'))
    return circuit

def medir(circuit):
    sim = cirq.Simulator()
    result = sim.run(circuit, repetitions=1)
    bits = result.measurements["m"][0]
    return "".join(str(b) for b in bits)

# ⚙️ Chamada ao nucleo HARPIA externo : Simbiotic AI which resolves decoherence control
def calcular_F_opt(H, S, C, I, T):
    try:
        result = subprocess.run(
            ["./sphy_simbiotic_entangle_ai", str(H), str(S), str(C), str(I), str(T)],
            capture_output=True, text=True, check=True
        )
        match = re.search(r"([-+]?\d*\.\d+|\d+)", result.stdout)
        if match:
            return float(match.group(0))
        else:
            raise ValueError(f"❌ Falha ao extrair valor de saida do subprocesso. Saida: {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Erro ao executar o subprocesso: {e.stderr}", file=sys.stderr)
        raise
    except FileNotFoundError:
        print("\n❌ Erro: Executavel './sphy_simbiotic_entangle_ai' nao encontrado.", file=sys.stderr)
        print("Certifique-se de que o arquivo esta no diretorio correto e tem permissao de execucao.", file=sys.stderr)
        raise

# 🔬 Funcao Worker para Simular um Unico Frame
def simulate_frame(frame_data):
    frame, num_qubits, noise_prob, initial_coherence = frame_data
    
    # Cada processo comeca com sua propria copia da coerencia inicial
    # e a evolui de forma independente
    sphy_coherence = initial_coherence
    
    # Garante que cada processo tenha uma semente de aleatoriedade unica
    random.seed(os.getpid() * frame)
    
    estados_ideais = ['0' * num_qubits, '1' * num_qubits]

    # --- Simulacao do Frame ---
    circuito = gerar_ghz_state(num_qubits, noise_prob)
    resultado = medir(circuito)

    # --- Calculo do HARPIA Core (autonomo em cada processo) ---
    # A coerencia aqui e local ao processo, nao compartilhada
    H = random.uniform(0.95, 1.0)
    S = random.uniform(0.95, 1.0)
    C = sphy_coherence / 100
    I = abs(H - S)
    T = frame

    try:
        boost = calcular_F_opt(H, S, C, I, T)
    except Exception as e:
        return None, None, f"\nErro critico ao calcular F_opt no frame {frame}: {e}"

    delta = boost * 0.7
    novo_coherence = min(100, sphy_coherence + delta)
    ativado = delta > 0

    aceito = (resultado in estados_ideais) and ativado
    
    # --- Geracao do Hash e Log ---
    dados_uid = f"{frame}{resultado}{novo_coherence}{H}{S}{C}{I}{T}{ativado}"
    uid_sha256 = hashlib.sha256(dados_uid.encode()).hexdigest()
    
    log_entry = [
        frame, resultado, round(H, 4), round(S, 4),
        round(C, 4), round(I, 4), round(boost, 4),
        round(novo_coherence, 4), "✅" if aceito else "❌",
        uid_sha256
    ]
    
    # Retorna os dados do frame e a coerencia final deste processo
    return log_entry, novo_coherence, None

# 🚀 Simulacao principal (agora com multiprocessamento)
def executar_simulacao(num_qubits, total=100000, noise_prob=0.3):
    print("=" * 60)
    print(f"    🧿 HARPIA QGHZ STABILIZER - {num_qubits} Qubits - {total:,} Frames")
    print("=" * 60)

    timecode = datetime.now().strftime("%Y%m%d_%H%M%S")
    nome_csv = os.path.join(LOG_DIR, f"qghz_{num_qubits}q_log_{timecode}.csv")
    nome_fig = os.path.join(LOG_DIR, f"qghz_{num_qubits}q_graph_{timecode}.png")

    # Listas para coletar os resultados de todos os processos
    log_data = []
    sphy_evolution = []
    validos = 0
    initial_coherence = 90.0 # NAO e mais um objeto compartilhado

    # Prepara os dados para cada frame a ser simulado
    # Cada processo recebe a mesma coerencia inicial
    frame_inputs = [(f, num_qubits, noise_prob, initial_coherence) for f in range(1, total + 1)]
    
    num_processes = os.cpu_count()
    print(f"🔄 Usando {num_processes} processos para simular...")
    
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap_unordered(simulate_frame, frame_inputs), total=total, desc="⏳ Simulando GHZ"))

    # Processa os resultados de forma centralizada apos a simulacao
    for log_entry, new_coherence, error in results:
        if error:
            print(f"\n{error}", file=sys.stderr)
            continue # Pula o processamento do resultado com erro

        if log_entry:
            log_data.append(log_entry)
            sphy_evolution.append(new_coherence)
            if log_entry[-2] == "✅":
                validos += 1

    acceptance_rate = 100 * (validos / total) if total > 0 else 0
    print(f"\n✅ GHZ States aceitos: {validos}/{total} | {acceptance_rate:.2f}%")

    with open(nome_csv, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Frame", "Result", "H", "S", "C", "I", "Boost", "SPHY (%)", "Accepted", "UID_SHA256"])
        writer.writerows(log_data)
    print(f"🧾 CSV salvo: {nome_csv}")

    plt.figure(figsize=(12, 5))
    plt.plot(range(1, len(sphy_evolution) + 1), sphy_evolution, color="darkcyan", label="⧉ SPHY Coherence")
    
    if log_data:
        scatter_colors = ['green' if row[-2] == "✅" else 'red' for row in log_data]
        plt.scatter(range(1, len(sphy_evolution) + 1), sphy_evolution,
                    c=scatter_colors, s=8, alpha=0.6)
    
    plt.axhline(90, color='gray', linestyle="dotted", linewidth=1, label="Threshold")
    plt.title(f"📡 HARPIA SPHY Evolution - {num_qubits} Qubits - {total:,} Frames")
    plt.xlabel("Frames")
    plt.ylabel("SPHY Coherence (%)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(nome_fig, dpi=300)
    print(f"📊 Grafico salvo como: {nome_fig}")
    plt.show()

if __name__ == "__main__":
    qubits, pares = entrada_parametros()
    executar_simulacao(num_qubits=qubits, total=pares, noise_prob=0.3)