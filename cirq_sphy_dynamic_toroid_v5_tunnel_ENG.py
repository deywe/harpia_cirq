# -*- coding: utf-8 -*-
# File: cirq_sphy_dynamic_toroid_v6_tunnel_ENG.py (CORRIGIDO)
# Purpose: QUANTUM TUNNELING + HARPIA (Cirq) + SPHY Toroidal Visualization 3D
# Autor: Deywe Okabe | Corrigido para Cirq API (all_operations)
# Repare: exige meissner_core.py no mesmo diretório

from meissner_core import meissner_correction_step
import cirq
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import os, sys, random, hashlib
from datetime import datetime
from multiprocessing import Pool, Manager
from tqdm import tqdm
from scipy.interpolate import griddata
import pandas as pd

# === CONFIG ===
LOG_DIR = "logs_harpia_tunneling_cirq"
os.makedirs(LOG_DIR, exist_ok=True)

GRID_SIZE = 2
# cria 4 qubits em grade 0..1 x 0..1
QUBITS = [cirq.GridQubit(i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE)]
NUM_QUBITS = len(QUBITS)
TARGET_QUBIT = QUBITS[0]

# === INPUT ===
def get_user_parameters():
    """Coleta e valida os parâmetros de simulação do usuário."""
    try:
        num_qubits = NUM_QUBITS
        print(f"🔢 Number of Qubits (Toroidal lattice {GRID_SIZE}x{GRID_SIZE}): {num_qubits}")
        total_frames = int(input("🔁 Total Tunneling Attempts (Frames) to simulate: "))
        barrier_strength_input = float(input("🚧 Barrier Strength (0.0 to 1.0): "))
        if not (0.0 <= barrier_strength_input <= 1.0):
            raise ValueError("Barrier Strength must be between 0.0 and 1.0.")
        # Mapeia 1.0 para Pi/2 (90 graus)
        barrier_strength_theta = barrier_strength_input * np.pi / 2
        
        num_cpus = os.cpu_count()
        user_processes = int(input(f"❓ How many CPU cores to use? (1-{num_cpus}): "))
        user_processes = max(1, min(user_processes, num_cpus))
        
        return num_qubits, total_frames, barrier_strength_theta, user_processes
    except ValueError as e:
        print(f"❌ Input error: {e}")
        sys.exit(1)

# === CIRCUIT (toroidal entanglement) ===
def generate_toroidal_circuit(barrier_theta, noise_angle):
    """Gera um circuito quântico toroidal 2x2 com CZ, barreira RZ e ruído Rx."""
    circuit = cirq.Circuit()
    # superposition inicial
    for q in QUBITS:
        circuit.append(cirq.H(q))
        
    # conexões toroidais (ciclo 0-1-3-2-0)
    connections_idx = [(0,1), (1,3), (3,2), (2,0)]
    for a, b in connections_idx:
        circuit.append(cirq.CZ(QUBITS[a], QUBITS[b]))
        
    # barrier (phase) via Rz
    circuit.append(cirq.rz(barrier_theta).on(TARGET_QUBIT))
    
    # aplica ruído de fase como rotações Rx para todos (simula perturbação SPHY)
    if abs(noise_angle) > 1e-12:
        for q in QUBITS:
            circuit.append(cirq.rx(noise_angle / 2).on(q))
            
    # medição final (todos os qubits)
    circuit.append(cirq.measure(*QUBITS, key='m'))
    return circuit

# === Helper: expectation <Z_k> from state vector ===
def pauli_z_expectations_from_statevector(state_vector, num_qubits):
    """Calcula a expectativa de Pauli-Z para cada qubit a partir do vetor de estado."""
    probs = np.abs(state_vector)**2
    expectations = np.zeros(num_qubits, dtype=float)
    nstates = probs.size
    for basis_index in range(nstates):
        p = probs[basis_index]
        for q in range(num_qubits):
            # bit 0 -> +1, bit 1 -> -1 (definição Pauli Z)
            bit = (basis_index >> q) & 1
            expectations[q] += p * (1.0 if bit == 0 else -1.0)
    return expectations

# === Single-frame simulation (worker) ===
def simulate_frame(frame_data):
    """Função worker para simular um frame com multiprocessamento."""
    frame, total_frames, noise_prob, sphy_coherence, barrier_theta = frame_data
    random.seed((os.getpid() << 16) ^ frame)
    simulator = cirq.Simulator()

    noise_angle = random.uniform(-np.pi/8, np.pi/8) if random.random() < noise_prob else 0.0
    circuit = generate_toroidal_circuit(barrier_theta, noise_angle)

    # ⚠️ CORREÇÃO CRÍTICA: Filtra operações, não Momentos, para remover a medição.
    non_measurement_ops = [
        op for op in circuit.all_operations() 
        if not isinstance(op.gate, cirq.MeasurementGate)
    ]
    circuit_without_measure = cirq.Circuit(non_measurement_ops)
    
    try:
        # Simulação para obter o vetor de estado (e as fases)
        sim_result = simulator.simulate(circuit_without_measure)
        state_vector = sim_result.final_state_vector
        z_exps = pauli_z_expectations_from_statevector(state_vector, NUM_QUBITS)
        qubit_phases = [float(np.round(v, 4)) for v in z_exps]
    except Exception as e:
        return None, None, f"\nSimulation error in frame {frame}: {e}"

    # Simulação de Run para obter o resultado da medição (passou/não passou)
    try:
        run_result = simulator.run(circuit, repetitions=1)
        df = run_result.data
        if 'm' in df.columns:
            measured = df.iloc[0]['m']
        else:
            measured = df.iloc[0].tolist()[0]
    except Exception as e:
        return None, None, f"\nMeasurement error in frame {frame}: {e}"

    # Determina o resultado bruto (1 se qualquer qubit for 1, 0 caso contrário)
    try:
        bits = np.asarray(measured).astype(int).flatten()
        result_raw = int(bits.any())
    except Exception:
        result_raw = int(bool(measured))

    ideal_state = 1 # Tunelamento é considerado sucesso se o resultado for 1 (não-vazio)

    # === SPHY / Meissner feedback ===
    H = random.uniform(0.95, 1.0)
    S = random.uniform(0.95, 1.0)
    C = sphy_coherence / 100.0
    I = abs(H - S)
    T = frame
    psi_state = [3.0, 3.0, 1.2, 1.2, 0.6, 0.5]
    
    try:
        boost, phase_impact, psi_state = meissner_correction_step(H, S, C, I, T, psi_state)
    except Exception as e:
        return None, None, f"\nCritical error in Meissner step frame {frame}: {e}"

    delta = boost * 0.7
    new_coherence = min(100.0, sphy_coherence + delta)
    activated = delta > 0.0
    accepted = (result_raw == ideal_state) and activated

    ts = datetime.utcnow().isoformat()
    data_to_hash = f"{frame}:{result_raw}:{H:.4f}:{S:.4f}:{C:.4f}:{I:.4f}:{boost:.4f}:{new_coherence:.4f}:{ts}"
    sha = hashlib.sha256(data_to_hash.encode('utf-8')).hexdigest()

    # formata linha: Frame,Result,Q1_phase,...,H,S,C,I,Boost,SPHY(%),Accepted,SHA,Timestamp
    log_entry = [
        frame,
        result_raw,
        *qubit_phases,
        round(H,4),
        round(S,4),
        round(C,4),
        round(I,4),
        round(boost,4),
        round(new_coherence,4),
        "✅" if accepted else "❌",
        sha,
        ts
    ]
    return log_entry, new_coherence, None

# === 3D plot ===

def plot_3d_sphy_field(csv_filename, fig_filename_3d):
    """Plota a média das fases (expectativas Z) dos qubits 2x2 como um campo 3D interpolado."""
    try:
        df = pd.read_csv(csv_filename)
    except FileNotFoundError:
        print(f"❌ CSV not found: {csv_filename}")
        return
    
    phase_cols = [c for c in df.columns if c.startswith("Qubit_") and c.endswith("_Phase")]
    if len(phase_cols) < NUM_QUBITS:
        print("❌ CSV does not contain expected qubit phase columns.")
        return

    means = df[phase_cols].mean().values[:NUM_QUBITS]

    # Coordenadas (0,0), (0,1), (1,0), (1,1) para a grade 2x2
    X = np.array([0,0,1,1])
    Y = np.array([0,1,0,1])
    Z = means # Valor médio de Pauli-Z (fase)

    # Interpolação para suavizar o campo
    xi = np.linspace(X.min(), X.max(), 50)
    yi = np.linspace(Y.min(), Y.max(), 50)
    XI, YI = np.meshgrid(xi, yi)
    ZI = griddata((X, Y), Z, (XI, YI), method='cubic')

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 🎨 ESTILIZAÇÃO PARA COMBINAR COM O GRÁFICO DE EXEMPLO
    
    # Superfície
    surf = ax.plot_surface(XI, YI, ZI, cmap='viridis', edgecolor='k', linewidth=0.5, alpha=0.95) # Bordas escuras
    
    # Pontos de Dispersão (Qubits)
    ax.scatter(X, Y, Z, color='red', s=100, label='Avg Qubit Phase', alpha=0.8) # Pontos maiores e mais visíveis
    
    # Eixos e Títulos
    ax.set_xlabel('X (Lattice Position)')
    ax.set_ylabel('Y (Lattice Position)')
    ax.set_zlabel('Avg Pauli-Z (Phase Coherence)')
    ax.set_title('3D SPHY Shape Field (Toroidal Coherence Average)')
    
    # Limites dos Eixos para Corresponder ao Exemplo (especialmente o Z)
    # Ajuste o Z-lim dependendo da faixa esperada dos seus dados (Pauli-Z vai de -1 a 1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # Seus valores de Z (means) estão em torno de 0, então os limites do Z devem ser simétricos e pequenos.
    # Vou usar um exemplo de -0.1 a 0.1, mas você pode ajustar conforme seus dados reais.
    z_min, z_max = np.min(Z) - 0.05, np.max(Z) + 0.05
    if z_max < 0.1: # Garante que haja um mínimo de faixa visível
        z_max = 0.1
    if z_min > -0.1:
        z_min = -0.1
    ax.set_zlim(z_min, z_max) # Ajuste dinamicamente, mas com um mínimo/máximo
    
    # View angle (azimute e elevação) para corresponder ao exemplo
    ax.view_init(elev=20, azim=-60) # Ajuste estes valores se precisar de uma perspectiva diferente
    
    # Cores de fundo e grade
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.grid(True, linestyle='--', alpha=0.6) # Grade mais suave
    
    # Barra de cores
    fig.colorbar(surf, shrink=0.5, aspect=8, label='Avg Pauli-Z (Phase Coherence)')
    
    plt.savefig(fig_filename_3d, dpi=300)
    print(f"🖼️ 3D SPHY Shape Graph saved: {fig_filename_3d}")

# === main runner ===
def execute_simulation(total_frames, barrier_theta, num_processes, noise_prob=1.0):
    print("="*60)
    print(f" ⚛️ HARPIA-SPHY TOROIDAL: Quantum Tunneling • {total_frames:,} Frames")
    print(f" 🚧 Barrier Strength: {barrier_theta*180/np.pi:.2f}° RZ")
    print(f" 🔄 Using {num_processes} processes.")
    print("="*60)

    timecode = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(LOG_DIR, f"toroidal_{NUM_QUBITS}q_log_{timecode}.csv")
    fig_filename_3d = os.path.join(LOG_DIR, f"toroidal_{NUM_QUBITS}q_graph_3D_SPHY_{timecode}.png")

    manager = Manager()
    sphy_coherence = manager.Value('f', 90.0)
    log_data = manager.list()
    sphy_evolution = manager.list()
    valid_states = manager.Value('i', 0)

    # Frame data: frame, total_frames, noise_prob, sphy_coherence, barrier_theta
    frames = [(f, total_frames, noise_prob, sphy_coherence.value, barrier_theta) for f in range(1, total_frames+1)]

    with Pool(processes=num_processes) as pool:
        for log_entry, new_coherence, error in tqdm(pool.imap_unordered(simulate_frame, frames),
                                                    total=total_frames, desc="⏳ Simulating"):
            if error:
                print(error, file=sys.stderr)
                pool.terminate()
                break
            if log_entry:
                log_data.append(log_entry)
                sphy_evolution.append(new_coherence)
                sphy_coherence.value = new_coherence # update shared coherence
                if log_entry[-4] == "✅":
                    valid_states.value += 1

    total = total_frames
    accepted = valid_states.value
    acceptance_rate = 100.0 * accepted / total if total > 0 else 0.0
    mean_stability = np.mean(list(sphy_evolution)) if sphy_evolution else 0.0
    var_stability = np.var(list(sphy_evolution)) if sphy_evolution else 0.0

    print(f"\n✅ Tunneling Success Rate (SPHY Filtered): {accepted}/{total} | {acceptance_rate:.2f}%")
    print(f"📊 Average Stability Index (SPHY): {mean_stability:.6f}")
    print(f"📊 Stability Variance Index: {var_stability:.6f}")

    # write csv header
    qubit_phase_headers = [f"Qubit_{i+1}_Phase" for i in range(NUM_QUBITS)]
    header = ["Frame", "Result", *qubit_phase_headers, "H", "S", "C", "I", "Boost", "SPHY (%)", "Accepted", "SHA256_Signature", "Timestamp"]
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(list(log_data))
    print(f"🧾 CSV saved: {csv_filename}")

    # 3D plot
    plot_3d_sphy_field(csv_filename, fig_filename_3d)


if __name__ == "__main__":
    num_qubits, total_frames, barrier_theta, num_processes = get_user_parameters()
    execute_simulation(total_frames, barrier_theta, num_processes, noise_prob=1.0)