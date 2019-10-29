from tools.processing_tools import *
from tools.load_tools import *
import novainstrumentation as ni
from GrammarofTime.SSTS.backend.gotstools import *
from PDF_generator.reportGen import Report


example_path = "D:/PhD/Code/Hui_SuperProject/Data_Examples/"

signal = loadH5(example_path+"arthrokinemat_2018_06_02_23_51_55.h5")

fs = 1000
b = 10
ch1 = signal[b*fs:, 5]

# ch1 = (ch1-np.mean(ch1))/np.std(ch1)
ch1_pp = pre_processing(ch1, "M")

plt.plot(ch1_pp[0])
plt.show()

proc_ch1 = NewWindowStat(ch1, ["std", "mean"], 1000, 250)

plt.plot(proc_ch1)
plt.show()
