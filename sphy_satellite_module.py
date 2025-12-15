# -*- coding: utf-8 -*-
# ───────────────────────────────────────────────────────────────
# File: sphy_satellite_module.py
# Purpose: Simulação do campo de harmonia para integração quântica
# ───────────────────────────────────────────────────────────────

import numpy as np

def campo_de_harmonia_vibracional(pos):
    """
    Calcula um 'índice de harmonia' para uma posição 3D.
    Valores altos indicam maior harmonia.
    """
    x, y, z = pos
    harmonia = np.sin(x * 0.1) + np.cos(y * 0.2) + np.sin(z * 0.3)
    harmonia += 0.5 * (np.sin(x * 0.5) + np.cos(y * 0.6))
    return (harmonia + 3) / 6.0  # Normaliza entre 0 e 1

def simular_caminho_satelite(num_passos):
    """
    Simula o caminho de um satélite buscando harmonia e retorna as posições.
    """
    posicao_inicial = np.array([10.0, 0.0, 0.0])
    velocidade_inicial = np.array([0.0, 0.8, 0.5])
    
    posicoes = [posicao_inicial]
    pos_atual = posicao_inicial
    vel_atual = velocidade_inicial
    
    for _ in range(num_passos - 1):
        # 1. Simula a gravidade
        distancia = np.linalg.norm(pos_atual)
        aceleracao_grav = -pos_atual / distancia**3
        
        # 2. Sintonia com a Harmonia (usa um fator_sintonia)
        delta = 0.01
        direcao_harmonia = np.array([
            campo_de_harmonia_vibracional(pos_atual + [delta, 0, 0]) - campo_de_harmonia_vibracional(pos_atual),
            campo_de_harmonia_vibracional(pos_atual + [0, delta, 0]) - campo_de_harmonia_vibracional(pos_atual),
            campo_de_harmonia_vibracional(pos_atual + [0, 0, delta]) - campo_de_harmonia_vibracional(pos_atual)
        ])
        
        # Normaliza o impulso de sintonia
        if np.linalg.norm(direcao_harmonia) > 0:
            impulso_sintonia = direcao_harmonia / np.linalg.norm(direcao_harmonia) * 0.02
        else:
            impulso_sintonia = np.array([0, 0, 0])
        
        # 3. Atualiza a velocidade e a posição
        nova_vel = vel_atual + aceleracao_grav + impulso_sintonia
        nova_pos = pos_atual + nova_vel
        
        pos_atual, vel_atual = nova_pos, nova_vel
        posicoes.append(pos_atual)
        
    return np.array(posicoes)