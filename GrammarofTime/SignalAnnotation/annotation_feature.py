from tools.processing_tools import *
from tools.load_tools import *
import novainstrumentation as ni
from GrammarofTime.SSTS.backend.gotstools import *
from PDF_generator.reportGen import Report
from sklearn.neighbors import KernelDensity


example_path = "D:/PhD/Code/Hui_SuperProject/Data_Examples/"

signal = loadH5(example_path+"arthrokinemat_2018_06_02_23_51_55.h5")

fs = 1000
b = 10
ch1 = signal[b*fs:, 5]

# ch1 = (ch1-np.mean(ch1))/np.std(ch1)
ch1_pp = pre_processing(ch1, "M LP 2")
t = np.linspace(0, len(ch1_pp)/1000, len(ch1_pp))

hist, bins = np.histogram(ch1_pp, bins=6)
pd_s = prob_hist(ch1_pp, hist, bins, inverted=False, log=True)

doc = Report("libphys", "report_testWindows")

for windown_len in [10, 100, 250, 500, 1000]:
    proc_ch1 = NewWindowStat(ch1_pp, ["std", "mean", "zcr"], 1000, windown_len)
    proc_ch1_norm = stat_white(proc_ch1)

    total = np.sum(proc_ch1_norm, axis=1)

    doc.add_title(
        "Results with win_len="+str(windown_len)
    )
    fig = plt.figure()
    plt.plot(total*pd_s)
    plt.title("Test win_" + str(windown_len))

    doc.add_graph(fig, "Annotation of events with feature windows of "+str(windown_len), "annotation_testFeature&PD_win_"+str(windown_len))

doc.gen_pdf()

