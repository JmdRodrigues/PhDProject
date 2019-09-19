import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from GrammarofTime.SSTS.backend import gotstools as gt
from tools.string_plot_tools import plot_CharFreqTest, plotCharFreq, plot_TrendChar, plot_ProbChar
import scipy.stats as stats
from dtw import dtw, accelerated_dtw

import matplotlib.dates as mdates
import os

def jointProb_Segmented(time_all, serie1, serie2, cfg1, cfg2):

    #find when hpi is up
    matches = gt.ssts(serie1, cfg1)

    #find prob of mortgage
    matches2 = gt.ssts(serie2, cfg2, report="full")

    time, data = gt.Output1(matches2[1], matches, 1, time_all)

    ax1 = plt.subplot(2,1,1)
    ax2 = plt.subplot(2,1,2)

    ax1.plot(time_all, serie1)
    ax2.plot(time, data["p"])
    plt.show()




def jointProb(time_all, serie1, serie2, connotation):
    # overall data
    serie1_str, _ = gt.connotation([serie1], connotation)
    serie2_str, _ = gt.connotation([serie2], connotation)

    # get char distribution over a signal
    data_serie1 = gt.CharFreq(serie1_str, 1, 10)
    data_serie2 = gt.CharFreq(serie2_str, 1, 10)

    print(data_serie1["p"][:10])
    print(data_serie2["p"][:10])

    joint_p_prob = np.multiply(data_serie1["p"], data_serie2["p"])

    print(joint_p_prob)

    ax1 = plt.subplot(2,1,1)
    ax2 = plt.subplot(2,1,2)

    ax1.plot(time_all, serie1)
    ax2.plot(time_all, joint_p_prob)
    plt.show()


def plotCorrelation(time_all, hpi, mortgage, connotation):
    # overall data
    hpi_str, _ = gt.connotation([hpi], connotation)
    mortgage_str, _ = gt.connotation([mortgage], connotation)

    # get char distribution over a signal
    data_hpi = gt.CharFreq(hpi_str, 1, 10)
    data_mortgage = gt.CharFreq(mortgage_str, 1, 10)

    fig, axs = plt.subplots(3,3)
    fig.subplots_adjust(hspace=.5, wspace=.2)
    for i, k in enumerate(list(data_mortgage.keys())):
        print(k)
        print(i+1)

        for ii, kk in enumerate(list(data_mortgage.keys())):
            # d = gt.dtw_distance(data_hpi[k], data_mortgage[kk])
            cc = gt.cross_corr(data_hpi[k], data_mortgage[kk])
            # print(len(data_hpi[k]))
            print(time_all[0])
            time_all_cc = pd.to_datetime(np.linspace(pd.Timestamp(str(time_all[0])).value, pd.Timestamp(str(time_all[-1])).value, len(cc)))
            # r, p = stats.pearsonr(data_hpi[k], data_mortgage[kk])
            axs[i, ii].plot(time_all, data_mortgage[kk])
            axs[i, ii].plot(time_all, data_hpi[k])
            axs[i, ii].plot(time_all_cc, cc)
            axs[i, ii].set_title("("+k + "-" + kk + ")")

    plt.show()

def plotDistributionsPlots(time_all, hpi, mortgage, connotation):
    # overall data
    hpi_str, _ = gt.connotation([hpi], connotation)
    mortgage_str, _ = gt.connotation([mortgage], connotation)

    # get char distribution over a signal
    data_hpi = gt.CharFreq(hpi_str, 1, 10)
    data_mortgage = gt.CharFreq(mortgage_str, 1, 10)

    r, p = stats.pearsonr(data_hpi["p"], data_mortgage["p"])
    print("correlation test")
    print(r)
    print(p)

    # get distribution of char trends over the entire signal in differnt changes of the HPI index
    data_str_seas, names = gt.string_corr(hpi_str, mortgage_str)

    # get distributions of trends in the mortgage signal in time windows
    data_freq_win = gt.CharFreqTest(hpi_str, mortgage_str, "derivative", 10)

    # get distribution of chars in transitions
    data_str_seas_trans, names_trans = gt.string_corr_trans(hpi_str, mortgage_str)

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
    hpi_drop = np.array(hpi)- min(hpi)
    mortgage_drop = (mortgage - min(mortgage))
    ax1.plot(time_all, hpi_drop / max(hpi_drop), color="lightyellow", linewidth=2)
    ax1.set_title("HPI")
    ax1.set_xlabel("date (by month)")
    ax2.plot(time_all,  mortgage_drop / max(mortgage_drop), color="lightyellow", linewidth=2)
    ax2.set_xlabel("date (by quarter of year)")
    ax2.set_title("Mortgage")
    print(data_hpi)
    plot_TrendChar(data_hpi, time_all, ["p", "z", "n"], ax=ax1)
    plot_TrendChar(data_mortgage, time_all, ["p", "z", "n"], ax=ax2)
    plt.figure()
    plot_CharFreqTest(data_freq_win, "derivative", time_all)
    plotCharFreq(data_str_seas, ["p", "z", "n"], ax=plt.subplot(1, 2, 2))
    plt.show()

def plotDistributionProbPlots(time_all, hpi, mortgage, connotation):
    # overall data
    hpi_str, _ = gt.connotation([hpi], connotation)
    mortgage_str, _ = gt.connotation([mortgage], connotation)

    # get char distribution over a signal
    data_hpi = gt.CharFreq(hpi_str, 1, 10)
    data_mortgage = gt.CharFreq(mortgage_str, 1, 10)

    # get distribution of char trends over the entire signal in differnt changes of the HPI index
    data_str_seas, names = gt.string_corr(hpi_str, mortgage_str)

    # get distributions of trends in the mortgage signal in time windows
    data_freq_win = gt.CharFreqTest(hpi_str, mortgage_str, "derivative", 10)

    # get distribution of chars in transitions
    data_str_seas_trans, names_trans = gt.string_corr_trans(hpi_str, mortgage_str)

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
    hpi_drop = np.array(hpi) - min(hpi)
    mortgage_drop = (mortgage - min(mortgage))
    ax1.plot(time_all, hpi_drop / max(hpi_drop), color="lightyellow", linewidth=2)
    ax1.set_title("HPI")
    ax1.set_xlabel("date (by month)")
    ax2.plot(time_all, mortgage_drop / max(mortgage_drop), color="lightyellow", linewidth=2)
    ax2.set_xlabel("date (by quarter of year)")
    ax2.set_title("Mortgage")
    print(data_hpi)
    plot_ProbChar(data_hpi["p"], time_all, ax=ax1)
    plot_ProbChar(data_mortgage["p"], time_all, ax=ax2)
    plt.figure()
    # plot_CharFreqTest(data_freq_win, "derivative", time_all)
    # plotCharFreq(data_str_seas, ["p", "z", "n"], ax=plt.subplot(1, 2, 2))
    # plt.show()


#Hpi values
hp_index2 = pd.read_excel(r"C:\Users\Wolfgang\PycharmProjects\ppp\GrammarofTime\Trendalyze\HPI_PO_monthly_hist.xls", header=None, skiprows=5)

#selection of columns for time and hpi values
time_hpi = hp_index2[0]
hpi_values = hp_index2[19]

#load mortgage data (time and counts)
time = np.loadtxt(r"C:\Users\Wolfgang\PycharmProjects\ppp\GrammarofTime\Trendalyze\month.txt").astype(int)
mortgage = np.loadtxt(r"C:\Users\Wolfgang\PycharmProjects\ppp\GrammarofTime\Trendalyze\mortgage_list.txt")

#Count mortgage values per month
unique_mortgage = []
dates = []

for date in np.unique(time):
    unique_mortgage.append(np.sum(mortgage[np.where(time==int(date))[0]]))
    dates.append(datetime(year=int(str(date)[:4]), month=int(str(date)[4:6]), day=1))

#order time values, and select only the periods in common between HPI and mortgage
time_all = list(set(time_hpi).intersection(set(dates)))
time_all.sort()

print(np.where(time_hpi==time_all[0])[0])
hpi_values = list(hpi_values)[np.where(time_hpi==time_all[0])[0][0]:np.where(time_hpi==time_all[-1])[0][0]+1]

connotation = "D1 0.01"

hpi_seasonal = gt.RemLowPass(np.array(hpi_values), 10)
mortgage_lf = gt.RemLowPass(np.array(unique_mortgage), 10)
mortgage_seasonal = unique_mortgage - mortgage_lf
plt.plot(mortgage_lf)
plt.plot(mortgage_seasonal)
plt.show()
hpi_non_seas = gt.smooth(np.array(hpi_values), 25)
unique_mortgage = gt.smooth(np.array(unique_mortgage), 6)

hpi_seas_str, _ = gt.connotation([hpi_seasonal], connotation)
hpi_non_seas_str, _ = gt.connotation([hpi_non_seas], connotation)
mortgage_str, _ = gt.connotation([unique_mortgage], connotation)
mortgage_lf_str, _ = gt.connotation([mortgage_lf], connotation)

data_str_seas, names = gt.string_corr(hpi_seas_str, mortgage_str)
data_str_seas_trans, names_trans = gt.string_corr_trans(hpi_seas_str, mortgage_str)
data_freq_test = gt.CharFreqTest(hpi_seas_str, mortgage_str, "derivative", 10)


print(data_freq_test["p"])
print(data_freq_test["n"])
print(data_freq_test["z"])
print(data_str_seas_trans)
print(names_trans)


# plotCorrelation()
#
# plotCorrelation(time_all, hpi_seasonal, unique_mortgage, connotation)
#
# plotCorrelation(time_all, hpi_non_seas, unique_mortgage, connotation)
# jointProb(time_all, hpi_seasonal, unique_mortgage, connotation)

jointProb_Segmented(time_all, hpi_values, unique_mortgage, {
  "pre_processing": "",
  "connotation": "D1 0.01",
  "expression": "p+"
},{
  "pre_processing": "HP 10",
  "connotation": "D1 0.01",
  "expression": "p+"
})

plotDistributionProbPlots(time_all, hpi_seasonal, unique_mortgage, connotation)
# plotDistributionsPlots(time_all, hpi_values, unique_mortgage, connotation)
plotDistributionsPlots(time_all, hpi_non_seas, mortgage_seasonal, connotation)
# plotDistributionsPlots(time_all, hpi_seasonal, unique_mortgage, connotation)
# plotDistributionsPlots(time_all, hpi_non_seas, unique_mortgage, connotation)
# plotDistributionsPlots(time_all, unique_mortgage, hpi_seasonal, connotation)
# plotDistributionsPlots(time_all, unique_mortgage, hpi_non_seas, connotation)

# plotCharFreq(data, labels)
# plt.legend()
# plt.show()
# [ax.bar(time_all, data_str_seas[key]["p"], data_str_seas[key]["z"], data_str_seas[key]["n"], labels=['p','z','n']) for key in data_str_seas]



fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True)
ax1.plot(time_all, hpi_values)
ax1.set_title("HPI")
ax1.set_xlabel("date (by month)")
ax2.plot(time_all, unique_mortgage)
ax2.set_xlabel("date (by quarter of year)")
ax2.set_title("Mortgage")

# data = gt.CharFreq(mortgage_str, 1, 20)

# bars = [x+y for x, y in zip(data["p"], data["z"])]


#
# p_bar = ax3.bar(time_all, data["p"], 100,
#                 label="p")
# z_bar = ax3.bar(time_all, data["z"], 100,
#                 label="z", bottom=data["p"])
# n_bar = ax3.bar(time_all, data["n"], 100,
#                 label="n", bottom=bars)

plt.show()