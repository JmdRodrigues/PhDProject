import h5py
import numpy as np

def loadH5(file):

    dataFile = h5py.File(file, "r")
    dataSet = dataFile["data"][:]
    dataFile.close()
    signal = dataSet

    return signal