import numpy as np
import quaternion
import skinematics as skin
from scipy.integrate import cumtrapz
import math
import os

#Quaternion analysis

def ApplyComplementaryFilter(time, acc, gyro, mag, alpha):
    """
    @brief: Sensor fusion algorithm to translate IMUs and Smartphone data to Earth's reference frame
    :param time:
    :param acc: array, shape(N, 3)
    :param gyro: array, shape(N, 3)
    :param mag: array, shape(N, 3)
    :param alpha: filter gain
    :return: quaternion describing rotations of reference frame
    """

    numData = len(acc)
    quatFinal = []

    for ii in range(numData):
        accelVec = acc[ii, :]
        magVec = mag[ii, :]
        angvelVec = gyro[ii, :]

        # normalized acc vector
        accelVec_n = accelVec/(np.linalg.norm(accelVec))
        #normalized mag vector
        magVec_n = magVec/(np.linalg.norm(magVec))

        # earth coordinates: ENU (East, North, Up) configuration
        east = np.cross(magVec_n, accelVec_n)
        east /= np.linalg.norm(east)
        north = np.cross(accelVec_n, east)
        north /= np.linalg.norm(north)



        # Rotation Matrix
        basisVectors = np.vstack((east, north, accelVec_n))

        # Reference quaternion: acc + mag information
        quatRef = quaternion.quaternion(quaternion.from_rotation_matrix(basisVectors))


        if ii==0:
            quatFinal.append(quaternion.as_float_array(quatRef))
        else:
            if np.isnan(angvelVec / np.linalg.norm(angvelVec)).any():
                gyroVec_n = angvelVec
            else:
                gyroVec_n = angvelVec / np.linalg.norm(angvelVec)

            dt = (time[ii] - time[ii-1])

            #gyroscope quaternion
            theta = (np.linalg.norm(angvelVec)*dt)/2

            quatUpdate = np.array(
                [np.cos(theta), gyroVec_n[0]*np.sin(theta), gyroVec_n[1]*np.sin(theta),
                 gyroVec_n[2]*np.sin(theta)]
            )

            #Final quaternion: acc+mag+gyro information
            quatFinal.append(
                quaternion.as_float_array(quaternion.slerp(quatRef,
                                                           quaternion.quaternion(
                                                               *quatFinal[ii-1])*quaternion.quaternion(
                                                               *quatUpdate), 0, 1, alpha
                                                           )
                                          )
            )

    return np.array(quatFinal)

def calculateQuaternion(time, acc, gyro, mag, cfg, posvel=False):
    """

    :param time:
    :param acc:
    :param gyro:
    :param mag:
    :param cfg: dic. keys: sensor fusion method (string); values: parameters (list)
        Supported sensor fusion methods and parameters
            "Complementary": [alpha(0,1)]
            "Madgwick":[q_init(list), beta(0,1)]
            "Analytical":[q_init (list, shape(4)), initial_pos (list, shape(3))]
    :param posvel: boolean. If True return position and velocity
    :return: quaternion array, shape(N,4) with configuration (w,x,y,z). If posvel = True return also position and velocity
    """

    #gravity reference
    reference = [0,0,-1]
    dt = float(1/np.mean(1./(np.diff(time))))
    rate = float(1/dt)

    try:
        method = list(cfg.keys())[0]
        print("Sensor fusion method: ", method)
    except:
        print("Check your inputs")

        if(type(cfg) != dict):
            print("Input parameter - cfg - does not match the requirements.")

        if np.shape(gyro)[1] != 3:
            axes = np.shape(gyro)[1]
            print("Excpected 3 axes for the gyroscope. Received " + str(axes) + "instead.")
        else:
            if(np.shape(acc) != np.shape(gyro)) or (np.shape(mag) != np.shape(gyro)):
                print("Sensors do not have the same shape.")

        if(len(time)!= len(acc)):
            print("Your time array does not match the sensor length.")

    if method != 0:
        if method =="Complementary":
            par_alpha = list(*cfg.values())[0]
            q = ApplyComplementaryFilter(time, acc, gyro, mag, alpha=par_alpha)

        elif method == "Madgwick":
            par_qinit = list(*cfg.values())[0]
            par_beta = list(*cfg.values())[1]
            # q = applyMadgwick(par_qinit, par_beta, acc, gyro, mag, dt)

        elif method == "Analytical":
            par_qinit = skin.quat.convert(list(*cfg.values())[0], to="rotmat")
            par_initialpos = list(*cfg.values())[1]
            q = skin.imus.analytical(par_qinit, gyro, par_initialpos, acc, rate)

        else:
            print("Unknown sensor fusion method.")

        #If you also want to estimate te position and velocity
        if(len(q) != 0) and posvel:
            #scikit kinematics implementation
            accReSensor = acc - skin.vector.rotate_vector(reference, skin.quat.q_inv(q))
            accReSpace = skin.vector.rotate_vector(accReSensor, q)

            # Position and velocity through integration, assuming 0-velocity at t=0
            # Assume thtat initial position is equal to gravity
            vel = np.nan*np.ones_like(accReSpace)
            pos = np.nan*np.ones_like(accReSpace)

            for ii in range(accReSpace.shape[1]):
                vel[:, ii] = cumtrapz(accReSpace[:, ii], dx=1./rate, initial=0)
                pos[:, ii] = cumtrapz(vel[:, ii], dx=1./rate, initial=reference[ii])

            return q, pos, vel
        return q

#Angular Tools

def calculateRelativeAngleSingleIMU(quat, direction="Y", reference=[0,0,-1]):
    """

    :param quat: array, shape(N,4)
    :param direction:
    :param reference:
    :return:
    """

    #direction quaternion based on input direction
    if direction=="X":
        dir_quat = [0, 1, 0, 0]
    elif direction=="Y":
        dir_quat = [0, 0, 1, 0]
    elif direction=="Z":
        dir_quat = [0, 0, 0, 1]
    else:
        #default
        dir_quat = [0, 0, 1, 0]

    num = len(quat)

    #pure quaternion rotation in sensor frame
    q = np.array(
        [skin.quat.q_mult(*skin.quat.q_mult(quat[i, :], dir_quat), skin.quat.q_conj(quat[i, :]))[0] for i in range(num)]
    )

    return skin.vector.angle(q[:, 1:], reference)*(180/math.pi)

def v_norm(v):
    """

    :param v:
    :return: mag of v
    """
    if(type(v) is list):
        v = np.array(v)

    return np.sqrt(np.sum(np.power(v, 2)))

def v_normalization(v):
    """

    :param v:
    :return: Normalized vector list
    """

    if(type(v) is list):
        v = np.array(v)
    norm = v_norm(v)
    return np.ndarray.tolist(v/norm)

def get_angle(v1, v2):
    """
    get unsigned angle (acos) between v1 and v2

    :param v1:
    :param v2:
    :return:
    """

    return np.arccos(np.dot(np.array(v1), np.array(v2)) / (v_norm(v1)*v_norm(v2))) * (180/math.pi)

def get_angle_plane_vector(vn, v):

    return np.arcsin(np.dot(np.array(vn), np.array(v)) / (v_norm(vn) * v_norm(v)))*(180/math.pi)

def get_directionvector_quat(quat, direction="Y"):
    #direction quaternion based on input direction
    if direction=="X":
        return quaternion.as_float_array(quaternion.quaternion(*quat) * quaternion.quaternion(*[0, 1, 0, 0])
                                         * quaternion.quaternion(*quat).conj())[1:]
    elif direction=="Y":
        return quaternion.as_float_array(quaternion.quaternion(*quat) * quaternion.quaternion(*[0, 0, 1, 0])
                                         * quaternion.quaternion(*quat).conj())[1:]
    elif direction=="Z":
        return quaternion.as_float_array(quaternion.quaternion(*quat) * quaternion.quaternion(*[0, 0, 0, 1])
                                         * quaternion.quaternion(*quat).conj())[1:]
    else:
        #default
        print("Unkwon direction")

def get_frontal_angles(quat1, quat2, CHEST = 1):
    """
    Get the angles between a device and the frontal anatomical plane

    :param quat1:
    :param quat2:
    :param CHEST:
    :return:
    """

    if CHEST == 1:
        return get_angle_plane_vector(get_directionvector_quat(quat2, direction="Y"), get_directionvector_quat(quat1, direction="Z"))
    elif CHEST == 2:
        return get_angle_plane_vector(get_directionvector_quat(quat1, direction="Y"), get_directionvector_quat(quat2, direction="Z"))

def get_sagital_angles(quat1, quat2, CHEST = 1):
    """
    Get the angles between a device and the sagital anatomical plane
    :param quat1:
    :param quat2:
    :param CHEST:
    :return:
    """
    if CHEST == 1:
        return get_angle_plane_vector(get_directionvector_quat(quat1, direction="X"), get_directionvector_quat(quat2))
    elif CHEST == 2:
        return get_angle_plane_vector(get_directionvector_quat(quat2, direction="X"), get_directionvector_quat(quat1))

def get_angle_imus(quat1, quat2, direction="Y"):
    """

    :param quat1:
    :param quat2:
    :param directions:
    :return:
    """

    return get_angle(get_directionvector_quat(quat1, direction), get_directionvector_quat(quat2, direction))
