# webots/C2_pClient_rl/C2_pClient_rl.py

import numpy as np
import os
import pickle
import struct
from controller import Robot

if not os.path.exists("backup"):
    os.makedirs("backup")

TRAINING_MODE = True
POPULATION_SIZE = 10 if TRAINING_MODE else 1
MUTATION_RATE = 0.02
MUTATION_SCALE = 0.05

class NeuralNetwork:
    def __init__(self, input_dim):
        self.input_size = input_dim
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

    def clone(self):
        c = NeuralNetwork(self.input_size)
        c.W1 = self.W1.copy()
        c.b1 = self.b1.copy()
        c.W2 = self.W2.copy()
        c.b2 = self.b2.copy()
        c.W3 = self.W3.copy()
        c.b3 = self.b3.copy()
        return c

    def mutate(self):
        # Agora mutamos as 3 camadas
        for p in [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]:
            mask = np.random.rand(*p.shape) < MUTATION_RATE
            # A escala da mutação é importante. 
            noise = np.random.randn(*p.shape) * MUTATION_SCALE
            p += mask * noise

    def save(self, filename="best_policy.pkl"):
        # Pequena correção para garantir que guarda no sitio certo
        if not filename.endswith(".pkl"): filename += ".pkl"
        
        with open(filename, "wb") as f:
            pickle.dump(self, f)
        print(f">>> POLICY GUARDADA: {filename} <<<")

    @staticmethod
    def load(path="best_policy.pkl"):
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    net = pickle.load(f)
                    print(">>> POLICY CARREGADA COM SUCESSO <<<")
                    return net
            except:
                print(">>> AVISO: Erro ao carregar Policy (Formato incompatível). Começando do zero.")
                return None
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

camera = robot.getDevice("camera")
camera.enable(timeStep)
width, height = camera.getWidth(), camera.getHeight()

dist_sensors = [robot.getDevice(f'ps{i}') for i in range(8)]
for s in dist_sensors:
    s.enable(timeStep)

# ======================================================
# HELPERS
# ======================================================
n_max = (1/4095)*100
n_min = (1/34)*100
def get_sensor_values():
    raw_values = np.array([max(s.getValue(),0.0001) for s in dist_sensors])
    normalized = (1/(raw_values))*100
    normalized = np.clip((normalized - n_min) / (n_max - n_min), 0.0, 1.0)
    return normalized

def get_camera_features():
    img = camera.getImage()
    if img is None: return np.zeros(3)
    x, y = width // 2, height // 2
    return np.array([
        camera.imageGetRed(img, width, x, y) / 255.0,
        camera.imageGetGreen(img, width, x, y) / 255.0,
        camera.imageGetBlue(img, width, x, y) / 255.0
    ])

# ======================================================
# REWARD FUNCTION SIMPLIFICADA
# ======================================================
def calculate_reward(sensors, vL, vR, score, prev_score, prev_max_sensor):
    done = False
    reward = 0.0

    # --- 1. LIMITES ---
    HARD_COLLISION = 0.95
    max_sensor = np.max(sensors)

    if max_sensor > HARD_COLLISION:
        return -100.0, True, max_sensor

    # --- 2. MOVIMENTOS ---
    linear = (vL + vR) / (2 * max_speed)
    angular = abs(vL - vR) / (2 * max_speed)

    # --- 3. LÓGICA "ANTI-PIÃO" (Reforçada) ---
    # Se ele está de costas para a parede (sensores baixos), não pode rodar!
    if max_sensor < 0.3:
        if angular > 0.1: 
            reward -= 2.0 
        if linear > 0.5:
            reward += 1.0 
            
    else:
        
        left_pressure = np.mean(sensors[5:8])  # Obstáculo à Esquerda -> Virar Direita
        right_pressure = np.mean(sensors[0:3]) # Obstáculo à Direita -> Virar Esquerda
        
        turning_left = vR > vL
        turning_right = vL > vR
        
        ACTIVATION_THRESHOLD = 0.4

        if right_pressure > ACTIVATION_THRESHOLD:
            if turning_left: 
                reward += right_pressure * 2.0 
            elif turning_right: 
                reward -= right_pressure * 5.0

        if left_pressure > ACTIVATION_THRESHOLD:
            if turning_right: 
                reward += left_pressure * 2.0
            elif turning_left: 
                reward -= left_pressure * 5.0

    # --- 4. DERIVADA ---
    diff = prev_max_sensor - max_sensor
    
    # Só recompensa derivada se estiver em perigo
    if prev_max_sensor > 0.4:
        if diff > 0.01: 
            reward += 1.0
        elif diff < -0.01:
            reward -= 0.5 

    # Velocidade Base
    if linear > 0: reward += 0.2
    else: reward -= 1.0

    # --- 5. CHECKPOINTS ---
    if score > prev_score:
        reward += 1000.0

    return reward, done, max_sensor

def request_reset():
    # Limpar buffer
    while receiver.getQueueLength() > 0: receiver.nextPacket()
    
    emitter.send(b"RESET")
    leftMotor.setVelocity(0)
    rightMotor.setVelocity(0)
    
    # Pequena pausa para garantir que o supervisor processa
    robot.step(timeStep) 
    return 0

# ======================================================
# EVOLUTION SETUP
# ======================================================
# Inputs: 8 sensores + 3 camara + 2 feedback motores = 13
INPUT_DIM = 8 + 8
population = [NeuralNetwork(INPUT_DIM) for _ in range(POPULATION_SIZE)]

best_policy = NeuralNetwork.load()
best_score = -1e9
best_fitness = -1e9

# Se já existe um bom, ele é o pai de todos, mas com mutações para não estagnar
if best_policy:
    population[0] = best_policy.clone()
    for i in range(1, POPULATION_SIZE):
        population[i] = best_policy.clone()
        population[i].mutate()

# ======================================================
# TRAINING LOOP
# ======================================================
policy_idx = 0
current_nn = population[policy_idx]
fitness = 0
steps = 0
previous_score = 0
no_progress_counter = 0

# Max steps por episódio
MAX_STEPS = 10000 
previous_dist = np.zeros(8) # Inicializa zerado

print(f"--- INICIANDO GERAÇÃO 1 / INDIVIDUO {policy_idx} ---")
max_sensor = 0.0
while robot.step(timeStep) != -1:
    
    # 1. Ler Sensores
    sensors = get_sensor_values() # 0 (longe) a 1 (perto)
    #cam = get_camera_features()
    # Feedback dos motores (normalizado)
    #motor_feedback = np.array([leftMotor.getVelocity(), rightMotor.getVelocity()]) / max_speed
    
    # 2. Rede Neural
    state = np.concatenate([sensors, previous_dist])
    outputs = current_nn.forward(state) # [-1, 1]
    
    # 3. Controlar Motores (MUDANÇA IMPORTANTE)
    # Permitir valores negativos para rotação no eixo
    vL = outputs[0] * max_speed
    vR = outputs[1] * max_speed
    
    leftMotor.setVelocity(vL)
    rightMotor.setVelocity(vR)
    
    # 4. Receber Score do Supervisor
    current_score = previous_score
    while receiver.getQueueLength() > 0:
        data = receiver.getBytes()
        if len(data) == 4:
            unpacked = struct.unpack('<i', data)[0]
            # Filtro simples para garantir que lemos scores válidos
            if unpacked >= current_score: 
                current_score = unpacked
        receiver.nextPacket()
    
    # 5. Calcular Fitness
    step_reward, done, max_sensor = calculate_reward(sensors, vL, vR, current_score, previous_score, max_sensor)
    fitness += step_reward
    steps += 1
    
    # Detecção de estagnação (se o score não subir durante muito tempo)
    if current_score > previous_score:
        no_progress_counter = 0
    else:
        no_progress_counter += 1
        
    if no_progress_counter > 800: # Se ficar 10s sem progresso
        done = True
        fitness -= 10 # Penalização por ficar parado
    
    previous_score = current_score
    previous_dist = sensors

    # 6. Fim do Episódio
    if done or steps > MAX_STEPS:
        if current_score < 60:
            fitness -= 10000
        # Normaliza fitness pelo tempo (opcional, mas ajuda a comparar passos curtos vs longos)
        print(f"Indivíduo {policy_idx} | Score: {current_score} | Fitness: {fitness:.2f}")
        saved = False # Flag para saber se guardamos
        
        # CASO 1: Bateu o recorde de distância (O MAIS IMPORTANTE)
        if current_score > best_score:
            best_score = current_score   # <--- FALTAVA ISTO!
            best_fitness = fitness       # Atualiza o fitness associado a este score
            
            best_policy = current_nn.clone()
            best_policy.save("best_policy.pkl")
            
            # Guardar backup histórico
            if current_score >= 50:
                best_policy.save(f"backup/best_policy_score_{current_score}_fit_{int(fitness)}.pkl")
                
            print(f"🏆 NOVO RECORDE ABSOLUTO: {best_score} pontos! (Fitness: {fitness:.2f})")
            saved = True
            
        # CASO 2: Chegou ao mesmo sítio, mas foi mais eficiente (Maior Fitness)
        elif current_score == best_score and fitness > best_fitness:
            best_fitness = fitness
            
            best_policy = current_nn.clone()
            best_policy.save("best_policy.pkl")
            
            # Opcional: Atualizar o backup se quiseres a versão mais eficiente desse score
            if current_score >= 50:
                best_policy.save(f"backup/best_policy_score_{current_score}_fit_{int(fitness)}.pkl")

            print(f"⚡ MELHORIA DE EFICIÊNCIA: Score igual ({best_score}), mas melhor Fitness.")
            saved = True
        
        # Próximo individuo
        policy_idx += 1
        
        # Se acabou a geração
        if policy_idx >= POPULATION_SIZE:
            print(f"\n=== NOVA GERAÇÃO (Melhor Fitness: {best_fitness:.2f}) ===")
            policy_idx = 0
            
            # Elitismo: O melhor mantém-se
            population[0] = best_policy.clone()
            
            # Os outros são mutações
            for i in range(1, POPULATION_SIZE):
                population[i] = best_policy.clone()
                population[i].mutate()
        # Preparar próximo
        current_nn = population[policy_idx]
        fitness = 0
        steps = 0
        previous_score = request_reset()
        no_progress_counter = 0
        previous_dist = np.zeros(8)
        max_sensor = 0.0