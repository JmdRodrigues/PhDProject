import numpy as np
import GrammarofTime.SSTS.backend.gotstools as ssts
import matplotlib.pyplot as plt
from operator import itemgetter
from datetime import datetime, timedelta
import pandas as pd
import itertools


month = np.loadtxt(r"C:\Users\Wolfgang\PycharmProjects\ppp\GrammarofTime\Trendalyze\month.txt").astype(int).astype(str)
mortgage = np.loadtxt(r"C:\Users\Wolfgang\PycharmProjects\ppp\GrammarofTime\Trendalyze\mortgage_list.txt")

#Load HP index data
hp_index = pd.read_csv(r"C:\Users\Wolfgang\PycharmProjects\ppp\GrammarofTime\Trendalyze\Data\HPI\HPI.csv", header=None, skiprows=1)
hp_index2 = pd.read_excel(r"C:\Users\Wolfgang\PycharmProjects\ppp\GrammarofTime\Trendalyze\Data\HPI_2\HPI_PO_monthly_hist.xls", header=None, skiprows=5)

time_hpi_1 = hp_index[0]
hpi_values1 = hp_index[1]


time_hpi_1 = [datetime(year=int(a.split('-')[0]), month=int(a.split('-')[1]), day=1) for a in time_hpi_1]
month = [datetime(year=int(a[0:4]), month=int(a[4:]), day=1) for a in month]

# print(month)

tuple_list = list(zip(mortgage, month))
tuple_list = [(sum(i[0] for i in group), key) for key, group in itertools.groupby(sorted(tuple_list, key = lambda i: i[1]), lambda i: i[1])]

m, t = ([i[0] for i in tuple_list], [i[1] for i in tuple_list])

time = sorted(list(set(time_hpi_1).intersection(t)))
hpi_values = hpi_values1[[time_hpi_1.index(x) for x in time]]


cfg = {
  "pre_processing": "",
  "connotation": "D1 0.01",
  "expression": "p+"
}

cfg2 = {
  "pre_processing": "HP 10",
  "connotation": "D1 0.01",
  "expression": "p+"
}

matches = ssts.ssts(hpi_values, cfg2)

matches2 = ssts.ssts(hpi_values1, cfg, report="full")


fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)

fig.suptitle("HPI and Mortgage")
axs[0].plot(time, hpi_values)
axs[1].set_xlabel("Date")
axs[0].set_ylabel("House Price Index")
[axs[0].axvspan(time[i[0]], time[i[1]-1], facecolor='green', alpha=0.10) for i in matches]
axs[1].plot(t, m)
axs[1].set_ylabel("Mortgage Counts")
[axs[1].axvspan(time[i[0]], time[i[1]-1], facecolor='green', alpha=0.10) for i in matches]
plt.show()