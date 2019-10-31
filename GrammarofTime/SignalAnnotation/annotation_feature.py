from tools.processing_tools import *
from tools.load_tools import *
import novainstrumentation as ni
from GrammarofTime.SSTS.backend.gotstools import *
from PDF_generator.reportGen import Report
from sklearn.neighbors import KernelDensity
from definitions import CONFIG_PATH
from tools.plot_tools import  plot_config, font_config, subplot_pars
from pandas import read_json

guide = CONFIG_PATH + "/Hui_SuperProject/MovDict.json"
guide_dict = read_json(guide).to_dict()
example_path = CONFIG_PATH + "/Hui_SuperProject/Data_Examples/"
#pdf_config
doc = Report("libphys", "report_AnnotationFiles")
for key in guide_dict.keys():

    signal = loadH5(example_path+guide_dict[key]["file"])


    fs = 1000
    b = 10
    # for i in range(5, 8):
    ch1 = signal[b * fs:(len(signal)//(guide_dict[key]["divider"]//2))+b*fs, 7]

    # ch1 = (ch1-np.mean(ch1))/np.std(ch1)
    ch1_pp = pre_processing(ch1, "M LP 2")
    t = np.linspace(0, len(ch1_pp) / 1000, len(ch1_pp))

    hist, bins = np.histogram(ch1_pp, bins=6)
    pd_s = prob_hist(ch1_pp, hist, bins, inverted=False, log=True)

    # plot config
    plot_config()
    font0, font1, font2 = font_config()

    # for windown_len in [10, 100, 250, 500, 1000]:
    windown_len = 500
    proc_ch1 = NewWindowStat(ch1_pp, ["std", "mean", "Azcr"], 1000, windown_len)
    proc_ch1_norm = stat_white(proc_ch1)

    total = np.sum(abs(proc_ch1_norm), axis=1)

    subplot_cenas = subplot_pars()

    fig = plt.figure(subplotpars=subplot_cenas)
    width, height = fig.get_size_inches()
    fig.set_size_inches([width, height*3])
    ax1 = plt.subplot(3,1,1)
    ax1.plot(proc_ch1_norm)

    ax2 = plt.subplot(3,1,2)
    ax2.plot(ch1/max(ch1))
    ax2.plot(pd_s)

    ax3 = plt.subplot(3,1,3)
    ax3.plot(total*pd_s)
    plt.title(guide_dict[key]["pattern"], fontproperties=font2)


    doc.add_title(
        "Results with win_len="+str(windown_len)
    )
    doc.add_graph(fig, "Annotation of events with features", "annotation_testFeature_"+guide[key]["file"])

print("...printing pdf")
doc.gen_pdf()

    # plt.show()



