import numpy as np
import matplotlib.pylab as plt
import pandas as pd

def plot_CharFreqTest(freq, connotation, time):
    if(connotation =="derivative"):
        p = plt.subplot(3,2,1)
        n = plt.subplot(3,2,3)
        z = plt.subplot(3,2,5)

        fr_p = np.array(freq["p"]).T
        fr_n = np.array(freq["n"]).T
        fr_z = np.array(freq["z"]).T

        p.stackplot(time, fr_p[0], fr_p[1], fr_p[2], labels=["p", "z", "n"])
        p.set_title("Trend of Mortgage when there is rising HPI")
        n.stackplot(time, fr_n[0], fr_n[1], fr_n[2], labels=["p", "z", "n"])
        n.set_title("Trend of Mortgage when there is a decreasing HPI")
        z.stackplot(time, fr_z[0], fr_z[1], fr_z[2], labels=["p", "z", "n"])
        z.set_title("Trend of Mortgage when there is stable HPI")
        z.set_xlabel("Time")
        plt.legend(loc='upper left')


def plotCharFreq(data, labels, ax):
    data["p"] = np.array(data["p"])/sum(data["p"])
    data["n"] = np.array(data["n"])/sum(data["n"])
    data["z"] = np.array(data["z"])/sum(data["z"])

    df = pd.DataFrame(data, index=list(data.keys())).T
    df.plot(kind="bar", stacked=False, ax=ax)


def plot_ProbChar(data, time, ax):
    ax.plot(time, data)
    plt.legend(loc="upper left")

def plot_TrendChar(data, time, labels,ax):
    ax.stackplot(time, data["p"], data["z"], data["n"], labels=labels, alpha=0.6)
    plt.legend(loc='upper left')