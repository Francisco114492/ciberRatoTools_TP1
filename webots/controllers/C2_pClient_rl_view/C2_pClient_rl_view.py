"""
C2_pClient_run.py - MODO DE EXECUÇÃO (SEM TREINO)
Carrega o 'best_policy.pkl' e corre infinitamente.
"""

import numpy as np
import os
import pickle
import struct
from controller import Robot

# ======================================================
# CLASSE DA REDE (TEM DE SER IGUAL À DO TREINO)
# ======================================================
class NeuralNetwork:
    def __init__(self, input_dim):
        self.input_size = input_dim
        # A MESMA ARQUITETURA QUE USASTE NO TREINO
        self.W1 = np.random.randn(input_dim, 64) * 0.5
        self.b1 = np.zeros(64)
        self.W2 = np.random.randn(64, 32) * 0.5
        self.b2 = np.zeros(32)
        self.W3 = np.random.randn(32, 2) * 0.5
        self.b3 = np.zeros(2)

    def forward(self, x):
        h1 = np.tanh(x @ self.W1 + self.b1)
        h2 = np.tanh(h1 @ self.W2 + self.b2)
        out = np.tanh(h2 @ self.W3 + self.b3)
        return out 

    # Função load estática para carregar o ficheiro
    @staticmethod
    def load(path="best_policy.pkl"):
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    net = pickle.load(f)
                    print(f">>> SUCESSO: '{path}' carregado! A iniciar demo... <<<")
                    return net
            except Exception as e:
                print(f"ERRO ao carregar pickle: {e}")
                return None
        else:
            print(f"ERRO: Ficheiro '{path}' não encontrado na pasta.")
            return None

# ======================================================
# ROBOT SETUP
# ======================================================
robot = Robot()
timeStep = int(robot.getBasicTimeStep())

receiver = robot.getDevice("receiver_robot")
receiver.enable(timeStep)
receiver.setChannel(1)

emitter = robot.getDevice("emitter_robot")
emitter.setChannel(2)

leftMotor  = robot.getDevice("left wheel motor")
rightMotor = robot.getDevice("right wheel motor")
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))

max_speed = 6.28
dist_sensors = [robot.getDevice(f'ps{i}') for i in range(8)]
for s in dist_sensors: s.enable(timeStep)

# ======================================================
# HELPERS (IGUAIS AO TREINO)
# ======================================================
n_max = (1/4095)*100
n_min = (1/34)*100

def get_sensor_values():
    raw_values = np.array([max(s.getValue(), 0.0001) for s in dist_sensors])
    normalized = (1/(raw_values))*100
    normalized = np.clip((normalized - n_min) / (n_max - n_min), 0.0, 1.0)
    return normalized

def request_reset():
    print(">>> RESET (Colisão ou Fim) <<<")
    while receiver.getQueueLength() > 0: receiver.nextPacket()
    emitter.send(b"RESET")
    leftMotor.setVelocity(0)
    rightMotor.setVelocity(0)
    # Pausa técnica para o Supervisor atuar
    for _ in range(10): robot.step(timeStep)
    return 0

# ======================================================
# MAIN LOOP DE EXECUÇÃO
# ======================================================

# 1. Carregar o Campeão
# Podes mudar o nome do ficheiro aqui se quiseres carregar um backup específico
# Ex: agent = NeuralNetwork.load("backup/best_policy_score_260_fit_4066.pkl")
agent = NeuralNetwork.load("best_policy.pkl")

if agent is None:
    print("A sair porque não há rede neural...")
    robot.step(timeStep)
    exit()

previous_dist = np.zeros(8)
previous_score = 0

print("--- INICIANDO RUN (DEMO MODE) ---")

while robot.step(timeStep) != -1:
    
    # 1. Sensores
    sensors = get_sensor_values()
    
    # 2. Decisão (Rede Neural)
    # Input Dim = 16 (8 atuais + 8 anteriores)
    state = np.concatenate([sensors, previous_dist])
    outputs = agent.forward(state)
    
    # 3. Atuação
    vL = outputs[0] * max_speed
    vR = outputs[1] * max_speed
    
    leftMotor.setVelocity(vL)
    rightMotor.setVelocity(vR)
    
    # 4. Atualizar Memória
    previous_dist = sensors
    
    # 5. Monitorizar Score (Só para print)
    while receiver.getQueueLength() > 0:
        data = receiver.getBytes()
        if len(data) == 4:
            score = struct.unpack('<i', data)[0]
            if score > previous_score:
                print(f"Checkpoint! Score: {score}")
                previous_score = score
        receiver.nextPacket()

    # 6. Detecção de Colisão (Para reiniciar automaticamente a demo)
    HARD_COLLISION = 0.95
    if np.max(sensors) > HARD_COLLISION:
        print(f"Batida detetada! Reiniciando a volta...")
        previous_score = request_reset()
        previous_dist = np.zeros(8)