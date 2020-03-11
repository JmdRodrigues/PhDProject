from tools.load_tools import load_sensor_IOTIP
from tools.alignment_tools import syncData, syncDevicesData
from EAWS4python.postureTools import calculateQuaternion, calculateRelativeAngleSingleIMU, get_frontal_angles
from novainstrumentation import lowpass, smooth
import matplotlib.pyplot as plt
import GrammarofTime.SSTS.backend.gotstools as gt

iotipsPath = "D:/PhD/Data/Day1/IMU"
iotip_wrist = "/IoTiP_FhP_Sensing_E2_5_E2_52_CA_76_DE_97"
iotip_hand = "/IoTiP_FhP_Sensing_F2_1_F2_11_B3_59_ED_0F"
iotip_elbow = "/IoTiP_Sensing_E4_13_E4_13_E7_E2_45_8A"
torso = "/Nexus5_358240053598702"

time = {}
acc_ = {}
gyr_ = {}
mag_ = {}

cfg1 = {
        "pre_processing":"S 100",
        "connotation": "A 0.7 D1 0.05",
        "expression": "[10]n(0z){1000,}"
    }

cfg2 = {
        "pre_processing":"S 100",
        "connotation": "A 0.7 D1 0.05",
        "expression": "[10].(1z){1000,}"
    }

i = 0
for tag, iotip, cfg, lag in zip(["wrist", "hand", "elbow", "torso"], [iotip_wrist, iotip_hand, iotip_elbow, torso], [cfg1, cfg1, cfg2, cfg1], [1, 0, 0, 0]):
    acc_[tag] = load_sensor_IOTIP(iotipsPath+iotip, "Green", "Accelerometer")
    gyr_[tag] = load_sensor_IOTIP(iotipsPath+iotip, "Green", "Gyroscope")
    mag_[tag] = load_sensor_IOTIP(iotipsPath+iotip, "Green", "Magnetometer")

    #Align Data
    time[tag], acc_[tag], gyr_[tag], mag_[tag] = syncData(acc_[tag], gyr_[tag], mag_[tag])

    matches = gt.ssts(acc_[tag][:10000, 0], cfg)
    delay = matches[lag][0]

    time[tag] = time[tag][delay:]
    acc_[tag] = acc_[tag][delay:]
    gyr_[tag] = gyr_[tag][delay:]
    mag_[tag] = mag_[tag][delay:]


#sync devices
time, acc_, gyr_, mag_ = syncDevicesData(acc_, gyr_, mag_)


#Pre_process
# acc = lowpass(acc_, f=25, fs=100)
# gyr = lowpass(gyr_, f=25, fs=100)
# mag = lowpass(mag_, f=25, fs=100)

cfg_complementary = {"Complementary":[0.98]}

q_qcf = {}

for segment in ["wrist", "hand", "elbow", "torso"]:
    q_qcf[segment] = calculateQuaternion(time, acc_[segment], gyr_[segment], mag_[segment], cfg=cfg_complementary)

ShoulderFlex = -1*get_frontal_angles(q_qcf["torso"], q_qcf["elbow"], CHEST=1)

"""
ShoulderFlexion --> [-1*get_frontal_angles(quaternion_torso, quaternion_arm, CHEST=1)]
ShoulderExtension ...
ShoulderAbduction --> 
"""

