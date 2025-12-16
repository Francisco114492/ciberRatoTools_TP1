"""my_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from tkinter import NO
import numpy as np
from controller import Robot

# create the Robot instance.
robot = Robot()

# Get simulation step length.
timeStep = int(robot.getBasicTimeStep())

# Constants of the e-puck motors and distance sensors.
cruiseVelocity = 5.0
num_dist_sensors = 8

# Get left and right wheel motors.
leftMotor = robot.getDevice("left wheel motor")
rightMotor = robot.getDevice("right wheel motor")

# Get frontal distance sensors.
dist_sensors = [
    robot.getDevice("ps" + str(x)) for x in range(num_dist_sensors)
]  # distance sensors
list(map((lambda s: s.enable(timeStep)), dist_sensors))  # Enable all distance sensors

# Disable motor PID control mode.
leftMotor.setPosition(float("inf"))
rightMotor.setPosition(float("inf"))


# Set the initial velocity of the left and right wheel motors.
leftMotor.setVelocity(cruiseVelocity)
rightMotor.setVelocity(cruiseVelocity)

camera = robot.getDevice("camera")
camera.enable(timeStep)
BLUE = "blue"
RED = "red"
YELLOW = "yellow"
lastColor = RED
colores = [RED, BLUE, YELLOW]
danger = True

near_front = 100
near_side = 90
import matplotlib.pyplot as plt


def changeColor(r, g, b):
    if r > g + b and r > 70:
        return RED
    if r < 40 and g < 40 and b > 200:
        return BLUE
    if r > 220 and g > 220 and b < 60:
        return YELLOW

    return None


timeDanger = robot.getTime()
while robot.step(timeStep) != -1:

    print(danger)
    print(robot.getTime())
    if danger:
        print(danger)
        print(robot.getTime())

        leftMotor.setVelocity(-cruiseVelocity)
        rightMotor.setVelocity(cruiseVelocity)
        if robot.getTime() > timeDanger + 3.0:
            danger = False
        continue
    image = camera.getImage()
    # print(image)
    # r = camera.imageGetRed(image, camera.getWidth(), x, y)
    # g = camera.imageGetGreen(image, camera.getWidth(), x, y)
    # b = camera.imageGetBlue(image, camera.getWidth(), x, y)
    w = camera.getWidth()
    h = camera.getHeight()
    image = camera.getImage()

    # img = np.frombuffer(image, np.uint8).reshape((h, w, 4))
    # img_rgb = img[:, :, :3][:, :, ::-1]
    # plt.imshow(img_rgb)
    # plt.axis("off")
    # plt.show(block=False)
    # plt.pause(0.001)
    # plt.clf()

    r = camera.imageGetRed(image, w, w // 2, h // 2)
    g = camera.imageGetGreen(image, w, w // 2, h // 2)
    b = camera.imageGetBlue(image, w, w // 2, h // 2)
    newColor = changeColor(r, g, b)
    if newColor is not None:
        old = colores.index(lastColor)
        newInx=(old-1)%3
        if colores[newInx]==newColor:
            danger=True
            timeDanger = robot.getTime()
        lastColor = newColor

    print(r, g, b)
    print(lastColor)
    print()

    dist_sensor_values = [g.getValue() for g in dist_sensors]
    right_sensors = dist_sensor_values[:3]
    left_sensors = dist_sensor_values[-3:]
    sentido_de_rotacao = 1 if sum(right_sensors) > sum(left_sensors) else -1
    # print(dist_sensor_values)

    if (
        dist_sensor_values[0] > near_front
        or dist_sensor_values[7] > near_front
        or dist_sensor_values[1] > near_front * 1.2
        or dist_sensor_values[6] > near_front * 1.2
    ):
        # print("rotate")
        leftMotor.setVelocity(-cruiseVelocity * sentido_de_rotacao)
        rightMotor.setVelocity(cruiseVelocity * sentido_de_rotacao)
    else:
        # print("go")
        leftMotor.setVelocity(cruiseVelocity)
        rightMotor.setVelocity(cruiseVelocity)
