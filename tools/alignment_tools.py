import numpy as np

def syncData(acc_, gyr_, mag_, fs = 100):

    ns2sec = 1e-9

    #Adjust sensor's timestamp
    t_acc = (acc_[:,0] - acc_[0,0]) * ns2sec
    t_gyro = (gyr_[:,0] - gyr_[0,0]) * ns2sec
    t_mag = (mag_[:,0] - mag_[0,0]) * ns2sec

    #Use the time array of the accelerometer
    time_vector = np.arange(t_acc[0], t_acc[-1], 1/fs)

    #Define initial and final reference instants
    t_i = np.max([t_acc[0], t_gyro[0], t_mag[0]])
    t_f = np.max([t_acc[-1], t_gyro[-1], t_mag[-1]])

    #Interpolation betweem the min and max duration of all streams
    time_vector = time_vector[np.argmin(np.abs(time_vector - t_i)):np.argmin(np.abs(time_vector-t_f))]

    #Interpolate data
    acc = np.array([np.interp(time_vector, t_acc, acc_[:, i]) for i in range(1, 4)]).T
    gyro = np.array([np.interp(time_vector, t_gyro, gyr_[:, i]) for i in range(1, 4)]).T
    mag = np.array([np.interp(time_vector, t_mag, mag_[:, i]) for i in range(1, 4)]).T

    return time_vector, acc, gyro, mag

def syncDevicesData(acc_, gyr_, mag_, fs = 100):

    ns2sec = 1e-9

    t_acc, acc = {}, {}
    t_gyro, gyro = {}, {}
    t_mag, mag = {}, {}
    #Adjust sensor's timestamp
    for segment in ["wrist", "hand", "elbow", "torso"]:
        t_acc[segment] = (acc_[segment][:,0] - acc_[segment][0,0]) * ns2sec
        t_gyro[segment] = (gyr_[segment][:,0] - gyr_[segment][0,0]) * ns2sec
        t_mag[segment] = (mag_[segment][:,0] - mag_[segment][0,0]) * ns2sec

    #Use the time array of the accelerometer of the wrist
    time_vector = np.arange(t_acc["wrist"][segment][0], t_acc["wrist"][-1], 1/fs)

    #Define initial and final reference instants
    t_i = np.max([t_acc["wrist"][0], t_gyro["wrist"][0], t_mag["wrist"][0]])
    t_f = np.max([t_acc["wrist"][-1], t_gyro["wrist"][-1], t_mag["wrist"][-1]])

    #Interpolation betweem the min and max duration of all streams
    time_vector = time_vector[np.argmin(np.abs(time_vector - t_i)):np.argmin(np.abs(time_vector-t_f))]

    #Interpolate data
    for segment in ["wrist", "hand", "elbow", "torso"]:
        acc[segment] = np.array([np.interp(time_vector, t_acc[segment], acc_[segment][:, i]) for i in range(1, 4)]).T
        gyro[segment] = np.array([np.interp(time_vector, t_gyro[segment], gyr_[segment][:, i]) for i in range(1, 4)]).T
        mag[segment] = np.array([np.interp(time_vector, t_mag[segment], mag_[segment][:, i]) for i in range(1, 4)]).T

    return time_vector, acc, gyro, mag