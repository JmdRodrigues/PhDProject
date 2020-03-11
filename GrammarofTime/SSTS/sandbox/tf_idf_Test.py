import time
from GrammarofTime.SSTS.sandbox.connotation_sandbox import AmplitudeTrans, AmpChange, D1Speed, SignConnotation, addArrayofStrings
from tools.plot_tools import *
from tools.load_tools import loadH5
from definitions import CONFIG_PATH
from pandas import read_json


def Connotation2(sig):
    # amp_level = AmplitudeTrans(sig, 2, string.ascii_uppercase, method="quantiles")

    t0 = time.time()
    ampdiff_str = AmpChange(sig, 0.75, "absolute")

    ax1 = plt.subplot(1, 1, 1)
    # plot_textcolorized(wave3, wave3_conc_str_tpl[2], ax1)
    plot_textcolorized(sig, ampdiff_str, ax1)
    plt.show()
    t1 = time.time()

    print("Done with ampdiff...")
    print("time: " + str(t1 - t0))
    speed_str = D1Speed(sig, 0.75)
    t2 = time.time()
    print("Done with diff...")
    print("time: " + str(t2-t1))
    sign_str = SignConnotation(sig)
    t3 = time.time()
    print("Done with sign...")
    print("time: " + str(t3 - t2))
    print("creating string...")
    wave_str = addArrayofStrings([sign_str, ampdiff_str, speed_str])

    print("Done")

    return wave_str

guide = CONFIG_PATH + "/Hui_SuperProject/MovDict.json"
guide_dict = read_json(guide).to_dict()
example_path = CONFIG_PATH + "/Hui_SuperProject/Data_Examples/"

key = 12
signal = loadH5(example_path + guide_dict[key]["file"])

