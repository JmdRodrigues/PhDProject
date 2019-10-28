from hmmlearn import hmm, base
import numpy as np
import math
import matplotlib.pyplot as plt
from tools.plot_tools import Cplot, Csubplot
from tools.processing_tools import WindowStat


def trainModelDerivative():
    X_train = []
    for i in range(10):
        sign_dif1 = np.random.choice([-1, 1])
        sign_dif2 = np.random.choice([-1, 1])
        len1 = np.random.randint(10, 100)
        len2 = np.random.randint(50, 150)
        s_i = np.cumsum(np.r_[sign_dif1*np.zeros(len1), 10*np.ones(len2), np.zeros(len1), -1*np.ones(len2)])
        x_i = [[s_i_j] for s_i_j in s_i]
        X_train.append(x_i)
    X_train = np.concatenate(X_train)
    model = hmm.GaussianHMM(n_components=3).fit(X_train)

    return model


def WindowModel(s, model, kind, window_len, window='hanning'):
    output = np.zeros(len(s))
    win = eval('np.' + window + '(window_len)')

    if s.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if s.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return s

    WinRange = int(window_len / 2)

    sig = np.r_[s[WinRange:0:-1], s, s[-1:len(s) - WinRange:-1]]

    for i in range(int(WinRange), len(sig) - int(WinRange)):
        if (kind == "mean"):
            s_m = WindowStat(sig[i - WinRange:WinRange + i], statTool="mean", fs=1, window_len=window_len // 2)
        #         m = [np.mean(sig[i - WinRange:WinRange + i])]
        elif (kind == "dif"):
            s_m = np.diff(sig[i - WinRange:WinRange + i])
            s_m = np.insert(s_m, 0, 0)

        X_train = np.array([[s_mi] for s_mi in s_m])

        output[i - int(WinRange)] = model.score(X_train)
    #         print(m)

    return output

#signal example:
lgth = 100

noise = np.random.normal(0, 1, lgth)

s0 = 10*noise*np.ones(lgth)
s1 = noise*np.ones(lgth)

s = np.r_[s0, s1, s0]

s_mean = WindowStat(s, "mean", fs=1, window_len=len(s)//5)
s_std = WindowStat(s, "std", fs=1, window_len=len(s)//5)

model = trainModelDerivative()

Z_mean = WindowModel(abs(s_mean), model, "dif", window_len=len(s)/2)
Z_std = WindowModel(abs(s_std), model, "dif", window_len=len(s)/2)
Z = Z_mean+Z_std

graphs = [[s, s_mean, s_std], [s, Z_mean, Z_std, Z]]

Csubplot(2, 1, graphs)


plt.show()
