"""
C2_RL_Agent.py - DQN corrigido (colisão, reward e estado)
"""
import struct
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pickle
from controller import Robot

TRAINING_MODE = True 
MAX_NO_PROGRESS = 800
# ==========================================
# 1. REDE NEURONAL (DQN)
# ==========================================
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# 2. AGENTE DQN
# ==========================================
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
        self.device = torch.device("cpu")

        self.model = DQN(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-4)
        self.criterion = nn.MSELoss()

        self.epsilon = 1.0 if TRAINING_MODE else 0.0 
        self.epsilon_min = 0.15
        self.epsilon_decay = 0.9995
        self.gamma = 0.99 

        self.memory = []
        self.batch_size = 128
        self.memory_capacity = 50000
        self.train_step_counter = 0

    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return torch.argmax(self.model(s)).item()

    def store_experience(self, s, a, r, ns, done):
        if len(self.memory) >= self.memory_capacity:
            self.memory.pop(0)
        self.memory.append((s, a, r, ns, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        s, a, r, ns, d = zip(*batch)

        s  = torch.FloatTensor(s).to(self.device)
        ns = torch.FloatTensor(ns).to(self.device)
        r  = torch.FloatTensor(r).to(self.device)
        a  = torch.LongTensor(a).unsqueeze(1).to(self.device)
        d  = torch.FloatTensor(d).to(self.device)

        q = self.model(s).gather(1, a).squeeze()

        with torch.no_grad():
            q_next = self.model(ns).max(1)[0]
            q_target = r + (1 - d) * self.gamma * q_next

        loss = self.criterion(q, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.train_step_counter += 1
            if self.train_step_counter % 500 == 0:
                print(f"epsilon: {self.epsilon:.4f} | memory: {len(self.memory)}")

    def save(self):
        torch.save(self.model.state_dict(), "dqn_model.pth")

    def save_memory(self):
        # Guarda apenas as últimas 30.000 experiências para o ficheiro não ser gigante
        with open("dqn_memory.pkl", "wb") as f:
            pickle.dump(self.memory[-30000:], f)
        print(">>> MEMÓRIA DE EXPERIÊNCIA GUARDADA EM DISCO <<<")

    def load(self):
        if os.path.exists("dqn_model.pth"):
            self.model.load_state_dict(torch.load("dqn_model.pth"))
            self.model.eval()
            print(">>> MODELO CARREGADO <<<")

# ==========================================
# 3. SETUP DO ROBÔ
# ==========================================
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

camera = robot.getDevice("camera")
camera.enable(timeStep)
width  = camera.getWidth()
height = camera.getHeight()

dist_sensors = [robot.getDevice(f'ps{i}') for i in range(8)]
for s in dist_sensors:
    s.enable(timeStep)

# ==========================================
# 4. AÇÕES E AGENTE
# ==========================================
CRUISE_SPEED = 5.0
actions = [
    (CRUISE_SPEED, CRUISE_SPEED),
    (CRUISE_SPEED*0.5, CRUISE_SPEED),
    (CRUISE_SPEED, CRUISE_SPEED*0.5),
    (-CRUISE_SPEED*0.5, CRUISE_SPEED),
    (CRUISE_SPEED, -CRUISE_SPEED*0.5),
]

agent = DQNAgent(state_dim=13, action_dim=len(actions))
if os.path.exists("dqn_model.pth"):
    agent.load()
    if not TRAINING_MODE:
        agent.epsilon = 0.0

previous_score = 0

def normalize_sensors(v):
    return [min(x / 4000.0, 1.0) for x in v]

def get_camera_features():
    img = camera.getImage()
    if img is None:
        return [0.0, 0.0, 0.0]

    x, y = width // 2, height // 2
    r = camera.imageGetRed(img, width, x, y)
    g = camera.imageGetGreen(img, width, x, y)
    b = camera.imageGetBlue(img, width, x, y)
    return [r/255.0, g/255.0, b/255.0]

# ==========================================
# 5. REWARD (CORRIGIDO)
# ==========================================
def calculate_reward(dist_features, action, score, prev_score):
    string = ""
    reward = -0.02
    done = False

    # 1. Bónus de Score (Prioridade Máxima)
    if score > prev_score:
        reward += 150.0
        string += "+150.0"

    prox_direita = max(dist_features[0], dist_features[1], dist_features[2])
    prox_esquerda = max(dist_features[5], dist_features[6], dist_features[7])
    
    threshold = 0.2 # Distância a partir da qual ele começa a "sentir" a parede

    # Se houver parede à DIREITA
    if prox_direita > threshold:
        if action in [1, 3]: # Ações de virar à ESQUERDA (1:suave, 3:forte)
            reward += prox_direita * 0.1  # Bónus: quanto mais perto da parede, mais ganha por virar
            # print("Boa! A fugir da parede da direita.")
        elif action in [2, 4]: # Se tentar virar para CIMA da parede
            reward -= prox_direita * 1.0  # Penalização pesada

    # Se houver parede à ESQUERDA
    if prox_esquerda > threshold:
        if action in [2, 4]: # Ações de virar à DIREITA (2:suave, 4:forte)
            reward += prox_esquerda * 0.1
            # print("Boa! A fugir da parede da esquerda.")
        elif action in [1, 3]:
            reward -= prox_esquerda * 1.0

    if action == 0: # Se não está a ir em frente (Ação 0)
        reward += 0.1

    # 2. Sensores e Colisão
    max_sensor = max(dist_features)
    if max_sensor > 0.35:
        reward -= 0.5
        #print("Close")
    elif max_sensor > 0.45:
        reward -= 150.0 # Penalização pesada
        string += "-150.0"
        done = True
        print(string)
        return reward, done

    vL, vR = actions[action]
    speed = (vL + vR) / 10.0 

    if speed > 0:
        reward += speed * 0.5
        string += f"+{speed*0.5:.2f}"
    else:
        reward -= 1.5
        string += "-1.5"

    if max_sensor > 0.2:
        reward -= (max_sensor ** 2) * 2.0
        string += f"-{(max_sensor ** 2) * 2.0:.2f}"

    return reward, done

def request_reset():
    print("Enviando RESET...")
    emitter.send("RESET\0".encode('utf-8'))
    leftMotor.setVelocity(0.0)
    rightMotor.setVelocity(0.0)
    return 0

# ==========================================
# 6. LOOP PRINCIPAL
# ==========================================
current_state = None
current_action = 0
no_progress_steps = 0

best_score = float('-inf')
no_improvement_resets = 0
total_reward_episodio = 0

while robot.step(timeStep) != -1:

    # Atuar
    vL, vR = actions[current_action]
    leftMotor.setVelocity(vL)
    rightMotor.setVelocity(vR)

    raw = [s.getValue() for s in dist_sensors]
    dist = normalize_sensors(raw)
    cam  = get_camera_features()
    vL_norm = leftMotor.getVelocity() / CRUISE_SPEED
    vR_norm = rightMotor.getVelocity() / CRUISE_SPEED
    next_state = dist + cam + [vL_norm, vR_norm]

    # Score
    score = previous_score

    if receiver.getQueueLength() > 0:
        while receiver.getQueueLength() > 0:
            data = receiver.getBytes()

            if len(data) == 4:
                try:
                    score = struct.unpack('<i', data)[0]
                except Exception as e:
                    print(f"Erro ao descompactar: {e}")
            receiver.nextPacket()

    if current_state is not None:
        reward, done = calculate_reward(dist, current_action, score, previous_score)
        total_reward_episodio += reward
        if score > previous_score:
            print(f"Prev score: {previous_score}; Curr score: {score}; Reward: {total_reward_episodio:.2f}; steps with no progress: {no_progress_steps}")
            no_progress_steps = 0
        else:
            no_progress_steps += 1

        if no_progress_steps % 100 == 0 and no_progress_steps != 0:
            print(f"Prev score: {previous_score}; Curr score: {score}; Reward: {total_reward_episodio:.2f}; steps with no progress: {no_progress_steps}")

        if TRAINING_MODE:
            agent.store_experience(current_state, current_action, reward, next_state, done)

        if done or (no_progress_steps > MAX_NO_PROGRESS):
            # Parar motores
            leftMotor.setVelocity(0.0)
            rightMotor.setVelocity(0.0)
            
            if TRAINING_MODE:
                print(f"Treinando... (Score Final: {score})")
                for _ in range(250):
                    agent.train()
                
                if score >= best_score:
                    best_score = score
                    agent.save()
                    agent.save_memory()
                    no_improvement_resets = 0
                    print(f">>> RECORDE BATIDO: {best_score}")
                else:
                    no_improvement_resets += 1
                    if no_improvement_resets >= 10:
                        print(">>> RECUPERANDO MODELO ANTERIOR...")
                        agent.load()
                        agent.epsilon = 0.5 
                        no_improvement_resets = 0
            
            previous_score = request_reset()
            total_reward_episodio = 0
            no_progress_steps = 0
            current_state = None
            continue


    previous_score = score
    current_action = agent.select_action(next_state, TRAINING_MODE)
    current_state = next_state
