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

hpi_values = list(hpi_values)[np.where(time_hpi==time_all[0])[0][0]:np.where(time_hpi==time_all[-1])[0][0]+1]

connotation = "D1 0.01"

hpi_seasonal = gt.RemLowPass(np.array(hpi_values), 10)
hpi_non_seas = gt.smooth(np.array(hpi_values), 25)
mortgage_lf = gt.RemLowPass(np.array(unique_mortgage), 10)
mortgage_seasonal = unique_mortgage - mortgage_lf

#joint probability for hpi and mortgage in a time window (for seasonal data?)

## calculate the string of hpi and mortgage
hpi_seas_str, _ = gt.connotation([hpi_seasonal], connotation)
hpi_non_seas_str, _ = gt.connotation([hpi_non_seas], connotation)
mortgage_str, _ = gt.connotation([unique_mortgage], connotation)
mortgage_lf_str, _ = gt.connotation([mortgage_lf], connotation)


##find matches of up in hpi intervals
cfg1={
    "pre_processing": "~ 10",
    "connotation": "D1 0.005",
    "expression": "p+"
}
matches_hpi = gt.ssts(hpi_seasonal, cfg1, 'clean')


## determinate the frequency of each character per window
data_freq_test = gt.CharFreq(hpi_seas_str, 1, 10)
data_freq_test2 = gt.CharFreq(mortgage_str, 1, 10)

p_joint_prob = np.multiply(data_freq_test["p"], data_freq_test2["p"])

ax1 = plt.subplot(2,1,1)
ax2 = plt.subplot(2,1,2)
# ax3 = plt.subplot(3,1,3)

ax1.plot(time_all, hpi_values)
[ax1.axvspan(time_all[i[0]], time_all[i[1]-1], facecolor='green', alpha=0.10) for i in matches_hpi]
ax1.set_title("HPI chart")
ax1.set_ylabel("Amplitude")
# ax1.set_xlabel("years")

ax2.plot(time_all, data_freq_test2["p"], label="probability of Mortgage increase")
[ax2.axvspan(time_all[i[0]], time_all[i[1]-1], facecolor='green', alpha=0.10) for i in matches_hpi]

ax2.plot(time_all, p_joint_prob, label="joint probability")
ax2.set_title("Probability chart")
ax2.set_xlabel("years")
ax2.set_ylabel("Probability [0-1]")
# [ax2.axvspan(time_all[i[0]], time_all[i[1]-1], facecolor='green', alpha=0.10) for i in matches_hpi]
ax2.legend()
plt.show()

## find the ratio of the probability of p combined and individual in each time window


#probability of positive mortgage within an increasing HPI
