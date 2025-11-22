import math
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

    # -------------------------------------
    # Atualização de movimento
    # -------------------------------------
    def motion_update(self, move):
        new_belief = [[0.0 for _ in range(self.cell_cols)] for _ in range(self.cell_rows)]
        di, dj = 0, 0
        if move == "N": di, dj = 2, 0
        elif move == "S": di, dj = -2, 0
        elif move == "E": di, dj = 0, 2
        elif move == "W": di, dj = 0, -2

        for i in range(0, self.cell_rows, 2):
            for j in range(0, self.cell_cols, 2):
                if self.labMap[i][j] != ' ':
                    continue
                ni, nj = i + di, j + dj
                # se não bate numa parede e é uma célula válida
                if 0 <= ni < self.cell_rows and 0 <= nj < self.cell_cols and self.labMap[ni][nj] == ' ':
                    new_belief[ni][nj] += self.belief[i][j]
                else:
                    # se não se move (bateu na parede)
                    new_belief[i][j] += self.belief[i][j]

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

        return self.cell_walls[ci][cj][dir]
    
    def sensor_model(self,measures,ang,point):
        prob=1
        for sensor_index,measure in enumerate(measures):
            has_wall=self.hasWall(sensor_index, ang, point)
            prob*=probabilidade_certo(measure,has_wall)
        return prob

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

def probabilidade_certo(metrica: float, has_wall: bool) -> float:
    if has_wall:
        if metrica > 140:
            return 0.98
        elif metrica >= 75:
            return 0.85
        else:
            return 0.40
    else:
        if metrica > 140:
            return 0.01
        elif metrica >= 75:
            return 0.40
        else:
            return 0.60
