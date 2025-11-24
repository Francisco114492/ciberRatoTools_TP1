import math
import joblib
import numpy as np


devices = {
    0: {"x": 0.030, "y": -0.010, "orientation": 1.27},
    1: {"x": 0.022, "y": -0.025, "orientation": 0.77},
    2: {"x": 0.000, "y": -0.031, "orientation": 0.00},
    3: {"x": -0.030, "y": -0.015, "orientation": 5.21},
    4: {"x": -0.030, "y": 0.015, "orientation": 4.21},
    5: {"x": 0.000, "y": 0.031, "orientation": 3.14159},
    6: {"x": 0.022, "y": 0.025, "orientation": 2.37},
    7: {"x": 0.030, "y": 0.010, "orientation": 1.87},
}

def load_model(path="model_wall.pkl"):
    return joblib.load(path)

class Belief:
    def __init__(self, labMap):
        """
        labMap: mapa carregado pela classe Map do my_controller
        """
        self.labMap = labMap
        self.lab_rows = len(labMap)        # ex: 13
        self.lab_cols = len(labMap[0])     # ex: 27
        # número de células (cada 2 posições do labMap correspondem a uma célula)
        self.cell_rows = (self.lab_rows + 1) // 2   # ex: 7
        self.cell_cols = (self.lab_cols + 1) // 2   # ex: 14
        self.cell_walls = [[{'N': False, 'S': False, 'E': False, 'W': False} 
                        for _ in range(self.cell_cols)] 
                        for _ in range(self.cell_rows)]
        self.belief = self._init_belief()
        self.model = load_model()
    def __str__(self):
        text = "Belief matrix:\n"
        for i in reversed(range(self.cell_rows)):
            text += " ".join(f"{self.belief[i][j]*100:.4f}" for j in range(self.cell_cols)) + "\n"
        return text
    
    
    # def __str__(self):
    #     """Representação detalhada com valores numéricos."""
    #     text = "Belief matrix (with map layout):\n"
    #     for i in reversed(range(self.cell_rows)):
    #         for j in range(self.cell_cols):
    #             if self.labMap[i][j] == ' ':
    #                 text += f"{self.belief[i//2][j//2]*100:5.2f} "
    #             else:
    #                 text += self.labMap[i][j] + "  "
    #         text += "\n"
    #     return text

    def __repr__(self):
        """Versão compacta: mostra intensidade com símbolos."""
        text = "Belief heatmap:\n"
        for i in reversed(range(0, self.cell_rows)):
            for j in range(0, self.cell_cols):
                p = self.belief[i][j]
                if p > 0.05:
                    text += "#"
                elif p > 0.01:
                    text += "+"
                else:
                    text += "."
            text += "\n"
        return text

    def _init_belief(self):
        cells = []
        # percorre índices de células (0..cell_rows-1), mapeando para labMap em li=i*2, lj=j*2
        for ci in range(self.cell_rows):
            for cj in range(self.cell_cols):
                li, lj = ci * 2, cj * 2
                if 0 <= li < self.lab_rows and 0 <= lj < self.lab_cols:
                    if self.labMap[li][lj] == ' ':
                        cells.append((ci, cj))
                        # Norte
                        if li == self.lab_rows - 1 or self.labMap[li+1][lj] in ['-', '+']:
                            self.cell_walls[ci][cj]['N'] = True
                        # Sul
                        if li == 0 or self.labMap[li-1][lj] in ['-', '+']:
                            self.cell_walls[ci][cj]['S'] = True
                        # Este
                        if lj == self.lab_cols - 1 or self.labMap[li][lj+1] in ['|', '+']:
                            self.cell_walls[ci][cj]['E'] = True
                        # Oeste
                        if lj == 0 or self.labMap[li][lj-1] in ['|', '+']:
                            self.cell_walls[ci][cj]['W'] = True

        if len(cells) == 0:
            raise ValueError("No free cells found in labMap")
        p = 1.0 / len(cells)
        belief = [[0.0 for _ in range(self.cell_cols)] for _ in range(self.cell_rows)]
        for (ci, cj) in cells:
            belief[ci][cj] = p
        return belief 

    def normalize(self):
        total = sum(sum(row) for row in self.belief)
        if total == 0:
            return
        for i in range(self.cell_rows):
            for j in range(self.cell_cols):
                self.belief[i][j] /= total


    def probabilidade_certo(self,metrica, has_wall):
        prob_parede = self.model.predict_proba([[metrica]])[0][1]

        if has_wall:
            return prob_parede
        else:
            return 1 - prob_parede

    # -------------------------------------
    # Atualização de movimento
    # -------------------------------------
    def motion_update(self, move):
        """
        move ∈ { 'N','S','E','W' }
        Usa cell_walls para determinar se a transição é possível.
        """
        new_belief = [[0.0 for _ in range(self.cell_cols)] for _ in range(self.cell_rows)]

        # deltas no espaço de células → 1 célula por movimento
        if move == "N":  dci, dcj = 1, 0
        elif move == "S": dci, dcj = -1, 0
        elif move == "E": dci, dcj = 0, 1
        elif move == "W": dci, dcj = 0, -1
        else:
            raise ValueError("Movimento inválido: " + move)

        for ci in range(self.cell_rows):
            for cj in range(self.cell_cols):

                p = self.belief[ci][cj]
                if p == 0:
                    continue  # nada para mover

                # há parede na direção do movimento?
                if self.cell_walls[ci][cj][move]:
                    # movimento impossível → probabilidade fica na mesma célula
                    new_belief[ci][cj] += p
                    continue

                # caso contrário tenta mover
                nci, ncj = ci + dci, cj + dcj

                if 0 <= nci < self.cell_rows and 0 <= ncj < self.cell_cols:
                    new_belief[nci][ncj] += p
                else:
                    new_belief[ci][cj] += p

        self.belief = new_belief
        self.normalize()

    # -------------------------------------
    # Atualização de medição
    # -------------------------------------
    def measurement_update(self, measures,ang):
        new_belief = [[0.0 for _ in range(self.cell_cols)] for _ in range(self.cell_rows)]
        for ci in range(self.cell_rows):
            for cj in range(self.cell_cols):
                li, lj = ci * 2, cj * 2 

                if self.labMap[li][lj] != ' ':
                    continue
                
                prob = self.sensor_model(measures,ang,(ci,cj))
                new_belief[ci][cj] = self.belief[ci][cj] * prob

        self.belief = new_belief
        self.normalize()

    def hasWall(self, sensor_index, ang, point):

        ci, cj = point
        # ang=0
        theta_sensor = (ang + devices[sensor_index]["orientation"]) % (2*math.pi)

        # N: 45° a 135°, E: 315°-45°, S: 225°-315°, W: 135°-225°
        theta_deg = math.degrees(theta_sensor) % 360

        if 45 <= theta_deg < 135:
            dir = 'N'
        elif 135 <= theta_deg < 225:
            dir = 'W'
        elif 225 <= theta_deg < 315:
            dir = 'S'
        else:
            dir = 'E'
        if point==(0,0):
            print(sensor_index,dir,math.degrees(ang))    
        return self.cell_walls[ci][cj][dir]
    
    def sensor_model(self,measures,ang,point):
        # prob_list=[]
        prob=1
        for sensor_index,measure in enumerate(measures):
            has_wall=self.hasWall(sensor_index, ang, point)
            prob*=max(probabilidade_certo(measure,has_wall),0.1)
        return prob
        #     if True:
        #         prob2=max(probabilidade_certo(measure,has_wall),0.1)
        #         if point==(0,0):
        #             print("Sensor",prob2,has_wall)  
        #         prob_list.append(prob2)
        # prob=sum(prob_list)/len(prob_list)

    def most_probable_cell(self):
        max_p = 0
        pos = (0, 0)
        for i in range(self.cell_rows):
            for j in range(self.cell_cols):
                if self.belief[i][j] > max_p:
                    max_p = self.belief[i][j]
                    pos = (i, j)
        return pos
    
    def print_cell_walls(self, ci=None, cj=None):
        if ci is not None and cj is not None:
            walls = self.cell_walls[ci][cj]
            print(f"Célula ({ci},{cj}): N={walls['N']} S={walls['S']} E={walls['E']} W={walls['W']}")
        else:
            print("Estado das paredes de todas as células:")
            for i in reversed(range(self.cell_rows)):
                for j in range(self.cell_cols):
                    w = self.cell_walls[i][j]
                    s = ""
                    s += "N" if w['N'] else "."
                    s += "S" if w['S'] else "."
                    s += "E" if w['E'] else "."
                    s += "W" if w['W'] else "."
                    print(f"({i},{j}):{s}", end="  ")
                print()
PROB_CONST=1146/8000
def probabilidade_certo(metrica: float, has_wall: bool) -> float:
    if has_wall:
        if metrica > 85:
            return 1-PROB_CONST
        else:
            return PROB_CONST
    else:
 
        if metrica >= 85:
            return 0
        else:
            return 1
