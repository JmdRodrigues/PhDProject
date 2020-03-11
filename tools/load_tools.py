import h5py
import numpy as np
import os

def load_npz(filepath):
    npz_file = np.load(filepath, allow_pickle=True)

    return npz_file

def load_npz_featuresHui(filepath):
    """
    0 EMG1

    1 EMG2

    2 EMG3

    3 EMG4

    4 Airborne Microphone

    5 Piezoelectric Microphone (Respiration) - Please don't regard this
    channel. It's not stable.

    6 ACC Upper X

    7 ACC Upper Y

    8 ACC Upper Z

    9 Goniometer X

    10 ACC Lower X

    11 ACC Lower Y

    12 ACC Lower Z

    13 Goniometer Y

    14 Gyro Upper X

    15 Gyro Upper Y

    16 Gyro Upper Z

    17 Force Sensor - Please don't regard this channel. It's breakable and
    the signal quality is worse.

    18 Gyro Lower X

    19 Gyro Lower Y

    20 Gyro Lower Z

    21 Pushbutton

    :param filepath:
    :return:
    """
    npz_f = load_npz(filepath)
    object_f = npz_f["arr"]

    return object_f

def loadH5(file):

    dataFile = h5py.File(file, "r")
    dataSet = dataFile["data"][:]
    dataFile.close()
    signal = dataSet

    return signal

def load_sensor_IOTIP(path, station_type = "Green", sensor_type="Accelerometer"):
    """

    :param path: folder directory
    :param sensor_type: type of sensor to load
    :return: array, shape(N,5)
    """

    iotip_folders = os.listdir(path)

    for folder in iotip_folders:
        if(station_type in folder):
            return np.loadtxt(path+'/'+folder+'/'+sensor_type+'.txt', delimiter=",")
        else:
            print("No file found corresponding to this structure")
            print(iotip_folders)

