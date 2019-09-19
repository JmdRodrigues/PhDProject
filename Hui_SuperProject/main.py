from GrammarofTime.SSTS.backend import gotstools as gt
from tools.load_tools import loadH5
from Hui_SuperProject.parser_reader import read_string
from tools.processing_tools import *
from tools.string_processing_tools import *
import matplotlib.pyplot as plt
from novainstrumentation import niplot, multilineplot
from tools.plot_tools import *
import regex as re

from novainstrumentation import sumvolve
from novainstrumentation.freq_analysis import fundamental_frequency

example_path = r"C:\Users\Wolfgang\PycharmProjects\ppp\Hui_SuperProject\Data_Examples\\"

signal = loadH5(example_path+"arthrokinemat_2018_06_02_23_51_55.h5")

fs = 1000
b = 10
ch1 = signal[b*fs:, 5]

#pre process
ch1 = smooth(mean_norm(ch1),100)
plt.plot(ch1)
plt.show()
#ch1 = smooth(ch1, 250)

#string for my event search
s = "(c ci i ic) * 20 c"
pattern, divisions = read_string(s)
divider = np.linspace(1, len(ch1), divisions).astype(int)

cfg1={
        "pre_processing": "",
        "connotation": "D1 0.05",
        "expression": "p+"
}

# str_d = gt.DiffC(ch1, 0.01)
# str_d_norm, indxs = consecutiveCount(str_d, type='0')
# freqs = CountSequences(str_d_norm, ["pn", "pz", "zn", "zp", "np", "nz"])
#
# win_string = CountSeqLeveled(str_d_norm, ["pn", "pz", "zn", "zp", "np", "nz"], 4)
# t_indxs = findStringIndexes(str_d_norm, win_string, indxs)
#
#
# plt.plot(WindowString(str_d_norm, win_string, "levenshtein", window_len=2*len(win_string)))
# plt.show()

# plt.plot(ch1)
# [plt.axvline(i) for i in t_indxs]


# sm_meanwave, meanwave = automeanwave(ch1, 1000)
#
# ax1 = plt.subplot(2,1,1)
# ax2 = plt.subplot(2,1,2)
# ax1.plot(meanwave)
# ax2.plot(ch1)
# ax2.plot(sm_meanwave)
# ax2.plot(np.diff(sm_meanwave))

# plt.show()

plt.plot(ch1)

# multilineplot(ch1, len(ch1)//9)
niplot()
plt.show()



for i in range(1, int(divisions)):
    sig = ch1[(divider[i-1]):divider[i]]
    sig_std = smooth(WindowStat(sig, "std", fs, int((len(ch1)/int(divisions))/100)), int((len(ch1)/int(divisions))/20))
    sig_sum = smooth(WindowStat(sig, "sum", fs, int((len(ch1)/int(divisions))/100)), int((len(ch1)/int(divisions))/20))
    sig_zcr = smooth(WindowStat(sig, "Azcr", fs, int((len(ch1)/int(divisions))/100)), int((len(ch1)/int(divisions))/20))

    # sig_pks = smooth(WindowStat(ch1, "findPks", fs, 500), 500)
    sig_ad = smooth(WindowStat(sig, "AmpDiff", fs, int((len(ch1)/int(divisions))/20)), int((len(ch1)/int(divisions))/20))

    final = sig_ad+sig_std+sig_sum+sig_zcr

    cfg1={
        "pre_processing": "",
        "connotation": "D1 0.05",
        "expression": "p+"
    }

    cfg2={
        "pre_processing": "",
        "connotation": "D1 0.05",
        "expression": "[pz]n"
    }

    cfg3={
        "pre_processing": "",
        "connotation": "D1 0.1",
        "expression": "n[zp]"
    }

    cfg4 = {
        "pre_processing": "",
        "connotation": "D1 0.05",
        "expression": "z+"
    }

    print(final)
    #
    # matches1 = gt.ssts(final, cfg1)
    # matches2 = gt.ssts(final, cfg2)
    # matches3 = gt.ssts(final, cfg3)
    matches4 = gt.ssts(final, cfg4)


    plt.plot(sig)

    plt.plot(sig_std + 2)
    plt.plot(sig_sum + 4)
    plt.plot(sig_zcr + 6)
    plt.plot(sig_ad + 8)
    plt.plot(final + 10)

    # [plt.axvline(i[0] + (i[1]-i[0])//2) for i in matches1]
    # [plt.axvline(i[0] + (i[1]-i[0])//2) for i in matches2]
    # [plt.axvline(i[0] + (i[1]-i[0])//2) for i in matches3]
    [plt.axvline(i[0] + (i[1]-i[0])//2) for i in matches4]

    #[plt.axvspan(i[0]-2, i[1]+2, facecolor='green', alpha=1) for i in matches1]
    #[plt.axvspan(i[0]-2, i[1]+2, facecolor='red', alpha=1) for i in matches2]
    #[plt.axvspan(i[0]-2, i[1]+2, facecolor='blue', alpha=1) for i in matches3]
    # [plt.axvspan(i[0]-2, i[1]+2, facecolor='orange', alpha=1) for i in matches4]

    plt.show()