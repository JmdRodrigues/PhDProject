import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from GrammarofTime.SSTS.backend import gotstools as gt
import matplotlib.dates as mdates
import os

#Hpi values
hp_index2 = pd.read_excel(r"D:\PhD\Code\GrammarofTime\Trendalyze\HPI_PO_monthly_hist.xls", header=None, skiprows=5)

#selection of columns for time and hpi values
time_hpi = hp_index2[0]
hpi_values = hp_index2[19]

#load mortgage data (time and counts)
time = np.loadtxt(r"D:\PhD\Code\GrammarofTime\Trendalyze\month.txt").astype(int)
mortgage = np.loadtxt(r"D:\PhD\Code\GrammarofTime\Trendalyze\mortgage_list.txt")

#Count mortgage values per month
unique_mortgage = []
dates = []

for date in np.unique(time):
    unique_mortgage.append(np.sum(mortgage[np.where(time==int(date))[0]]))
    dates.append(datetime(year=int(str(date)[:4]), month=int(str(date)[4:6]), day=1))

#order time values, and select only the periods in common between HPI and mortgage
time_all = list(set(time_hpi).intersection(set(dates)))
time_all.sort()


#SSTS------------------------
#configure for hpi series
cfg1={
    "pre_processing": "RM 25",
    "connotation": "D1 0.05",
    "expression": "p+"
}
#configure for mortgage series (only to have derivative string)
cfg2={
    "pre_processing": "",
    "connotation": "D1 0.01",
    "expression": ""
}


print(np.where(time_hpi==time_all[0])[0])
hpi_values = list(hpi_values)[np.where(time_hpi==time_all[0])[0][0]:np.where(time_hpi==time_all[-1])[0][0]+1]

matches = gt.ssts(hpi_values, cfg1)

matches2 = gt.ssts(unique_mortgage, cfg2, report="full")


fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1)

ax1.plot(time_all, hpi_values)
[ax1.axvspan(time_all[i[0]], time_all[i[1]], facecolor='green', alpha=0.10) for i in matches]
ax1.set_title("HPI")
ax1.set_xlabel("date (by month)")
ax2.plot(time_all, unique_mortgage)

max_p = []
std_err = []
l = []
for i in range(0, 5):
    matches_2 = [(match0+i, match1+i) for match0, match1 in matches]
    [ax2.axvspan(time_all[i[0]], time_all[i[1]], facecolor='green', alpha=0.10) for i in matches_2]
    ax2.set_xlabel("date (months)")
    ax2.set_title("Mortgage")
    ax2.set_ylabel("Amplitude")

    time, data = gt.Output1(matches2[1], matches_2, 1, time_all)
    prob_p = np.mean(data["p"])
    #plot the probability of going up when mortgage goes up

    max_p.append(prob_p)
    l.append(i)
    std_err.append(np.std(data["p"]))


matches_2 = [(match0+l[np.argmax(max_p)], match1+l[np.argmax(max_p)]) for match0, match1 in matches]
matches_0 = [(match0, match1) for match0, match1 in matches]
time, data = gt.Output1(matches2[1], matches_2, 1, time_all)
time0, data0 = gt.Output1(matches2[1], matches_0, 1, time_all)
ax3.plot(time0["p"], data0["p"])
ax3.set_xlabel("date (years)")
ax3.set_title("Mortgage Increasing Probability")
ax3.set_ylabel("Probability")

plt.show()

ax11 = plt.subplot(2,1,1)
ax12 = plt.subplot(2,1,2)

ax11.plot(time["p"], data["p"])
ax11.set_xlabel("date (years)")
ax11.set_title("Probability of Increasing Mortgage (2 months after the HPI started increasing)")
ax11.set_ylabel("Probability")
ax12.bar(l, max_p, yerr=std_err)
ax12.set_xlabel("months after HPI started to increase")
ax12.set_ylabel("mean probability")
ax12.set_title("Mean and STD for each month after the HPI has started to increase")



# matrix_1 =gt.string_matrix(matches2[1], matches)
#
# bars = [x+y for x, y in zip(data["p"], data["z"])]
#
# p_bar = ax3.bar(time["p"], data["p"], 100,
#                 label="p")
# z_bar = ax3.bar(time["z"], data["z"], 100,
#                 label="z", bottom=data["p"])
# n_bar = ax3.bar(time["n"], data["n"], 100,
#                 label="n", bottom=bars)
#
#
#
#
plt.legend()
plt.show()

ax = plt.subplot(111)

# ax.imshow(matrix_1)
# ax.set_xticks(np.arange(len(matches)))
# ax.set_yticks(np.arange(len(matches)))
# ax.set_xticklabels(time["p"])
# ax.set_yticklabels(time["p"])
# plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
#
# plt.show()
#
#
# width_t, width = gt.numerical_matrix(unique_mortgage, matches, time_all, "time_width")
# ax1 = plt.subplot(2, 1, 1)
# ax2 = plt.subplot(2, 1, 2)
# ax1.plot(time_all, unique_mortgage)
# ax2.plot(width_t, width)
# [ax1.axvspan(time_all[i[0]], time_all[i[1]], facecolor='green', alpha=0.10) for i in matches]
# [ax2.axvspan(time_all[i[0]], time_all[i[1]], facecolor='green', alpha=0.10) for i in matches]
# plt.show()
#
# slope_t, slope_m = gt.numerical_matrix(unique_mortgage, matches, time_all, "slope")
#
# ax1 = plt.subplot(2,1,1)
# ax2 = plt.subplot(2,1,2)
# ax1.plot(time_all, unique_mortgage)
# ax2.plot(slope_t, slope_m)
# [plt.axvspan(time_all[i[0]], time_all[i[1]], facecolor='green', alpha=0.10) for i in matches]
# plt.show()
#
# matrix_2 = gt.numerical_matrix(unique_mortgage, matches, time_all, "dtw")
#
# ax = plt.subplot(111)
# ax.imshow(matrix_2)
# ax.set_xticks(np.arange(len(matches)))
# ax.set_yticks(np.arange(len(matches)))
# ax.set_xticklabels(time["p"])
# ax.set_yticklabels(time["p"])
# plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
#
# plt.show()
#
# ss = np.sum(matrix_2, axis=0)
# print("matrix2----------")
#
# print(ss)
# print(len(ss))
#
# ax1 = plt.subplot(3,1,1)
# ax2 = plt.subplot(3,1,2)
# ax3 = plt.subplot(3,1,3)
#
# ax1.plot(time_all, hpi_values)
# ax1.set_xlabel("date (by quarter of year)")
# ax1.set_title("HPI")
# ax2.plot(time_all, unique_mortgage)
# ax2.set_xlabel("date (by quarter of year)")
# ax2.set_title("Mortgage")
# tt = []
# for i, match in enumerate(matches):
#     tt.append(time_all[match[0]] + (time_all[match[1]] - time_all[match[0]]) / 2)
#
# ax3.plot(tt, ss)
# ax3.set_xlabel("date (by quarter of year)")
# ax3.set_title("DTW distance (the higher the distance the less similar are the matches)")
# plt.show()
