from tools.processing_tools import *
from tools.load_tools import *
import novainstrumentation as ni
from GrammarofTime.SSTS.backend.gotstools import *
from PDF_generator.reportGen import Report
from sklearn.neighbors import KernelDensity
from definitions import CONFIG_PATH
from tools.plot_tools import  plot_config, font_config

example_path = CONFIG_PATH + "/Hui_SuperProject/Data_Examples/"
signal = loadH5(example_path+"arthrokinemat_2018_06_02_23_51_55.h5")

fs = 1000
b = 10
ch1 = signal[b*fs:, 5]

# ch1 = (ch1-np.mean(ch1))/np.std(ch1)
ch1_pp = pre_processing(ch1, "M LP 2")
t = np.linspace(0, len(ch1_pp)/1000, len(ch1_pp))

hist, bins = np.histogram(ch1_pp, bins=6)
pd_s = prob_hist(ch1_pp, hist, bins, inverted=False, log=True)

#plot config
plot_config()
font0, font1, font2 = font_config()

# for windown_len in [10, 100, 250, 500, 1000]:
windown_len = 10
proc_ch1 = NewWindowStat(ch1_pp, ["std", "mean", "zcr"], 1000, windown_len)
proc_ch1_norm = stat_white(proc_ch1)

total = np.sum(abs(proc_ch1_norm), axis=1)


fig = plt.figure()
plt.plot(total)
plt.plot(ch1_pp)
plt.plot(total*pd_s)
plt.title("Test win_" + str(windown_len), fontproperties=font2)
plt.show()



# #pdf_config
# doc = Report("libphys", "report_testWindows")
# doc.add_title(
#     "Results with win_len="+str(windown_len)
# )
# doc.add_graph(fig, "Annotation of events with feature windows of "+str(windown_len), "annotation_testFeature&PD_win_"+str(windown_len))
#
# print("...printing pdf")
# doc.gen_pdf()

