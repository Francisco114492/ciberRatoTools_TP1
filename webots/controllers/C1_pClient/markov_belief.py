import math


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
PROB_CONST=1146/8000

def calculate_probability(metrica: float, has_wall: bool) -> float:
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

class Belief:
    def __init__(self, labMap):
        """
        labMap: map loaded by the Map class of my_controller
        """
        self.labMap = labMap
        self.lab_rows = len(labMap)        # ex: 13
        self.lab_cols = len(labMap[0])     # ex: 27
        # number of cells (every 2 positions in the labMap correspond to one cell)
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
    
    
    def __repr__(self):
        """Compact version: shows intensity with symbols."""
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
        # traverses cell indices (0..cell_rows-1), mapping to labMap in li=i*2, lj=j*2
        for ci in range(self.cell_rows):
            for cj in range(self.cell_cols):
                li, lj = ci * 2, cj * 2
                if 0 <= li < self.lab_rows and 0 <= lj < self.lab_cols:
                    if self.labMap[li][lj] == ' ':
                        cells.append((ci, cj))
                        # North
                        if li == self.lab_rows - 1 or self.labMap[li+1][lj] in ['-', '+']:
                            self.cell_walls[ci][cj]['N'] = True
                        # South
                        if li == 0 or self.labMap[li-1][lj] in ['-', '+']:
                            self.cell_walls[ci][cj]['S'] = True
                        # East
                        if lj == self.lab_cols - 1 or self.labMap[li][lj+1] in ['|', '+']:
                            self.cell_walls[ci][cj]['E'] = True
                        # West
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
    # Movement update
    # -------------------------------------
    def motion_update(self, move):
        new_belief = [[0.0 for _ in range(self.cell_cols)] for _ in range(self.cell_rows)]

        if move == "N":  dci, dcj = 1, 0
        elif move == "S": dci, dcj = -1, 0
        elif move == "E": dci, dcj = 0, 1
        elif move == "W": dci, dcj = 0, -1
        else:
            raise ValueError("Invalid move: " + move)

        for ci in range(self.cell_rows):
            for cj in range(self.cell_cols):

                p = self.belief[ci][cj]
                if p == 0:
                    continue

                if self.cell_walls[ci][cj][move]:
                    new_belief[ci][cj] += p
                    continue

                nci, ncj = ci + dci, cj + dcj

                if 0 <= nci < self.cell_rows and 0 <= ncj < self.cell_cols:
                    new_belief[nci][ncj] += p
                else:
                    new_belief[ci][cj] += p

        self.belief = new_belief
        self.normalize()

    # -------------------------------------
    # Measurement update
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

        # N: 45° - 135°, W: 135°-225°, S: 225°-315°, E: 315°-360 and 0-45°
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
            prob*=max(calculate_probability(measure,has_wall),0.1)
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
    
    def print_cell_walls(self):
            print("Status of the walls of all cells:")
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
    def write_belief_in_file(self,filename):
        text = ""
        for i in reversed(range(self.cell_rows)):
            text += " ".join(f"{self.belief[i][j]:.3f}" for j in range(self.cell_cols)) + "\n"
        with open(filename,"a") as file:
            file.write(text+"\n")