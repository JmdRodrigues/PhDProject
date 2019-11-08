from tools.processing_tools import *
from tools.load_tools import *
import novainstrumentation as ni
from GrammarofTime.SSTS.backend.gotstools import *
from PDF_generator.reportGen import Report
from sklearn.neighbors import KernelDensity
from definitions import CONFIG_PATH
from tools.plot_tools import plot_config, font_config, subplot_pars, plot_textcolorized, strsignal2color
from pandas import read_json

guide = CONFIG_PATH + "/Hui_SuperProject/MovDict.json"
guide_dict = read_json(guide).to_dict()
example_path = CONFIG_PATH + "/Hui_SuperProject/Data_Examples/"
# pdf_config
doc = Report("libphys", "report_AnnotationFiles")

for key in guide_dict.keys():
    signal = loadH5(example_path + guide_dict[key]["file"])

    fs = 1000
    b = 10
    # for i in range(5, 8):
    ch1 = signal[b * fs:(len(signal) // (guide_dict[key]["divider"] // 4)) + b * fs, 7]
    ref = signal[b * fs:(len(signal) // (guide_dict[key]["divider"] // 4)) + b * fs, -1]
    # ch1 = (ch1-np.mean(ch1))/np.std(ch1)
    # ch1_pp = pre_processing(ch1, "M LP 2")
    ch1_pp = pre_processing(ch1, "M")
    ref = pre_processing(ref, "M")


    t = np.linspace(0, len(ch1_pp) / 1000, len(ch1_pp))

    hist, bins = np.histogram(ch1_pp, bins=6)
    pd_s = prob_hist(ch1_pp, hist, bins, inverted=False, log=True)

    # plot config
    plot_config()
    font0, font1, font2 = font_config()

    # for windown_len in [10, 100, 250, 500, 1000]:
    windown_len = 500

    proc_ch1 = NewWindowStat(ch1_pp, ["std", "mean", "Azcr"], 1000, windown_len)


    proc_str = []

    for nbr, feat in enumerate(proc_ch1.T):
        # print(feat)
        quantile_vals = np.quantile(feat, [0.25, 0.5, 0.75])

        str_feat = quantilstatesArray(feat, quantile_vals)

        plot_textcolorized(ch1_pp, str_feat, plt.subplot(1,1,1))
        plt.show()

    #     str_feat2 = quantilstatesArray(feat, quantile_vals, conc=False)
    #
    #     proc_str.append([str_feat2])
    #
    # proc_str = np.array(proc_str)
    # conc_proc_str = concat_np_strings(proc_str, 0)
    #
    # ax1 = plt.subplot(1, 1, 1)
    # ax1.plot(ref)
    # strsignal2color(ch1_pp, conc_proc_str[0], ax=ax1)
    # plt.show()








