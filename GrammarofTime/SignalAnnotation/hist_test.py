import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KernelDensity
from tools.PeakFinder import detect_peaks

def kde_sklearn(sig, x_samples, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(sig[:, np.newaxis])
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_samples[:, np.newaxis])
    return np.exp(log_pdf)


def plotwithhist(t, s, bins=50):
    data = {"t": t, "s": s}
    g = sns.JointGrid(x="t", y="s", data=data)
    sns.lineplot(t, s, ax=g.ax_joint)
    sns.distplot(s, bins=bins, kde=True, vertical=True, ax=g.ax_marg_y)


def dispHist(n, bins):
    h = max(n) - min(n)
    b = max(bins) - min(bins)

    return h, b


def Window_dist(s, fs=1, window_len=50, window='hanning'):
    output1 = np.zeros(len(s))
    output2 = np.zeros(len(s))
    output3 = np.zeros(len(s))

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
        print(i)
        sig_tmp = sig[i - WinRange:WinRange + i]
        x = np.linspace(min(sig_tmp) - 2, max(sig_tmp) + 2, 100)
        pdf = kde_sklearn(sig_tmp, x, bandwidth=1)
        # plt.plot(pdf)
        # plt.show()
        h, b = dispHist(x, pdf)
        output1[i - WinRange] = h
        output2[i - WinRange] = b
        output3[i - WinRange] = len(detect_peaks(pdf))


    return output1, output2, output3

"""Test 1"""
# test discovering states based on a sample observation
# generate new signal
lgth = 100

noise = np.random.normal(0, 1, lgth)

s0 = 10*noise*np.ones(lgth)
s1 = noise*np.ones(lgth)

s = np.r_[s0, s1, s0]

o1, o2, o3 = Window_dist(s, window_len=len(s)//10)

plt.plot(s)
plt.plot(o1)
plt.plot(o2)
plt.plot(o3)
plt.show()

"""Test 2"""
#test discovering states based on a sample observation
#generate new signal
time = np.linspace(0, 50, 1000)

s0 = 5*np.sin(40*time)
s1 = 5*np.sin(time)
s2 = 5*np.sin(10*time)

s = np.r_[s2, s1, s0, s1]

o1, o2, o3 = Window_dist(s, window_len=len(s)//10)

plt.plot(s)
plt.plot(o1)
plt.plot(o2)
plt.plot(o3)
plt.show()