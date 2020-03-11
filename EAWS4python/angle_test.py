import numpy as np
import quaternion
import math
import skinematics as skin
from EAWS4python import postureTools as pt
import novainstrumentation as ni

import matplotlib.pyplot as plt

import pandas as pd

def preprocessData(data):
    remove_lst = []
    for ii in range(len(data)):
        acc = data[ii, :3]
        gyr = data[ii, 3:6]
        mag = data[ii, 6:9]
        if(np.linalg.norm(acc) == 0 or np.linalg.norm(gyr) == 0 or np.linalg.norm(mag) == 0):
            remove_lst.append(ii)
    print(remove_lst)
    print(np.delete(data, remove_lst, 0))

    return np.delete(data, remove_lst, 0)

def preprocessSignal(signal):
    return ni.lowpass(signal, order = 1, fs=10, f=2)

#download data
data = pd.read_csv("D:/PhD/Code/EAWS4python/Data/TestPhone/180withY.csv", skiprows=3, delimiter=",").to_numpy()

data = preprocessData(data)

time = np.linspace(0, len(data)/10, len(data))
acc = data[:, :3]
# acc = preprocessSignal(acc)
# print(acc[0,:])
gyr = data[:, 3:6]
# gyr = preprocessSignal(gyr)
# print(gyr)
mag = data[:, 6:9]
# mag = preprocessSignal(mag)
# print(mag)
orientation = data[:, -2:]

plt.plot(acc)
plt.plot(gyr)
plt.plot(mag)
plt.show()

#define reference array
gravity = [0, 0, -1]
#define quaternion orientation of the sensors
#configure complementary filter
cfg_complemenatary = {"Complementary": [0.98]}
q_qcf = pt.calculateQuaternion(time, acc, gyr, mag, cfg=cfg_complemenatary)

angle_X = pt.calculateRelativeAngleSingleIMU(q_qcf, direction="X", reference= gravity)
angle_Y = pt.calculateRelativeAngleSingleIMU(q_qcf, direction="Y", reference= gravity)
angle_Z = pt.calculateRelativeAngleSingleIMU(q_qcf, direction="Z", reference= gravity)

angle = [angle_X, angle_Y, angle_Z]

tilt = [math.atan2(acc[i, 2], -acc[i, 1])*(180/math.pi) for i in range(np.shape(acc)[0])]


plt.plot(time, tilt, '-')
plt.plot(time, angle_Y, 'o')
plt.plot(time, angle_X, 'o')
plt.plot(time, angle_Z, 'o')
# for angle_i in angle:
#     plt.plot(time, angle_i, 'o')
plt.show()