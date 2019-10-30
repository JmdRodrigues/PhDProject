# -*- coding: utf-8 -*-

import sys
import json
import regex as re
import numpy as np

from scipy import signal, stats
from scipy.signal import filtfilt
from pylab import figure, plot, vlines
from dtw import dtw
from scipy.stats import ttest_ind

import matplotlib.pyplot as plt

from collections import Counter

with open(
        r'D:\PhD\Code\GrammarofTime\SSTS\backend\gots_dictionary.json') as data_file:
    gots_func_dict = json.load(data_file)


# Filtering methods
def smooth(input_signal, window_len=10, window='hanning'):
    """
    @brief: Smooth the data using a window with requested size.
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the beginning and end part of the output signal.
    @param: input_signal: array-like
                the input signal
            window_len: int
                the dimension of the smoothing window. the default is 10.
            window: string.
                the type of window from 'flat', 'hanning', 'hamming',
                'bartlett', 'blackman'. flat window will produce a moving
                average smoothing. the default is 'hanning'.
    @return: signal_filt: array-like
                the smoothed signal.
    @example:
                time = linspace(-2,2,0.1)
                input_signal = sin(t)+randn(len(t))*0.1
                signal_filt = smooth(x)
    @see also:  numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman,
                numpy.convolve, scipy.signal.lfilter
    @todo: the window parameter could be the window itself if an array instead
    of a string
    @bug: if window_len is equal to the size of the signal the returning
    signal is smaller.
    """

    if input_signal.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if input_signal.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return input_signal

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("""Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'""")

    sig = np.r_[2 * input_signal[0] - input_signal[window_len:0:-1],
                input_signal,
                2 * input_signal[-1] - input_signal[-2:-window_len - 2:-1]]

    if window == 'flat':  # moving average
        win = np.ones(window_len, 'd')
    else:
        win = eval('np.' + window + '(window_len)')

    sig_conv = np.convolve(win / win.sum(), sig, mode='same')

    return sig_conv[window_len: -window_len]


def RemLowPass(input_signal, window_len):
    """
    @brief: for a given signal input_signal, it removes the low frequency fluctuations.
    @params:
    input_signal: signal
    window_len: window to the signal to be removed
    """
    a = input_signal - smooth(input_signal, window_len=window_len)
    return a


def lowpass(s, f, order=2, fs=1000.0, use_filtfilt=True):
    """
    @brief: for a given signal s rejects (attenuates) the frequencies higher
    then the cuttof frequency f and passes the frequencies lower than that
    value by applying a Butterworth digital filter
    @params:
    s: array-like
    signal
    f: int
    the cutoff frequency
    order: int
    Butterworth filter order
    fs: float
    sampling frequency
    @return:
    signal: array-like
    filtered signal
    """
    b, a = signal.butter(order, f / (fs / 2))

    if use_filtfilt:
        return filtfilt(b, a, s)

    return signal.lfilter(b, a, s)


def highpass(s, f, order=2, fs=1000.0, use_filtfilt=True):
    """
    @brief: for a given signal s rejects (attenuates) the frequencies lower
    then the cuttof frequency f and passes the frequencies higher than that
    value by applying a Butterworth digital filter
    @params:
    s: array-like
    signal
    f: int
    the cutoff frequency
    order: int
    Butterworth filter order
    fs: float
    sampling frequency
    @return:
    signal: array-like
    filtered signal
    """

    b, a = signal.butter(order, f * 2 / (fs / 2), btype='highpass')
    if use_filtfilt:
        return filtfilt(b, a, s)

    return signal.lfilter(b, a, s)


def bandpass(s, f1, f2, order=2, fs=1000.0, use_filtfilt=True):
    """
    @brief: for a given signal s passes the frequencies within a certain range
    (between f1 and f2) and rejects (attenuates) the frequencies outside that
    range by applying a Butterworth digital filter
    @params:
    s: array-like
    signal
    f1: int
    the lower cutoff frequency
    f2: int
    the upper cutoff frequency
    order: int
    Butterworth filter order
    fs: float
    sampling frequency
    @return:
    signal: array-like
    filtered signal
    """
    b, a = signal.butter(order, [f1 * 2 / fs, f2 * 2 / fs], btype='bandpass')

    if use_filtfilt:
        return filtfilt(b, a, s)

    return signal.lfilter(b, a, s)


# Statistical methods
def stat_white(x):
    return (x - np.mean(x)) / np.std(x)


def normalization(signal, newMin, newMax, xmin=None, xmax=None):
    if xmin is None:
        xmin = np.min(signal)

    if xmax is None:
        xmax = np.max(signal)

    return (signal - xmin) * (newMax - newMin) / (xmax - xmin) + newMin


# Connotation Methods
def AmpC(s, t, p='>'):
    thr = ((np.max(s) - np.min(s)) * t) + np.min(s)
    if (p == '<'):
        s1 = (s <= (thr)) * 1
    elif (p == '>'):
        s1 = (s >= (thr)) * 1

    return s1


def DiffC(s, t, signs=['n', 'z', 'p']):
    # Quantization of the derivative.
    # TODO: Implement a better way of selecting chars
    ds1 = np.diff(s)
    x = np.empty(len(s), dtype=str)
    thr = (np.max(ds1) - np.min(ds1)) * t
    x[np.where(ds1 <= -thr)[0]] = signs[0]
    x[np.where(np.all([ds1 <= thr, ds1 >= -thr], axis=0))[0]] = signs[1]
    x[np.where(thr <= ds1)[0]] = signs[2]
    x[-1] = x[-2]

    return x


def Diff2C(s, t, symbols=['n', 'z', 'p']):
    # Quantization of the derivative.
    # TODO: Implement a better threshold methodology.
    dds1 = np.diff(np.diff(s))
    x = np.empty(len(s), dtype=str)
    thr = (np.max(dds1) - np.min(dds1)) * t
    x[np.where(dds1 <= -thr)[0]] = symbols[0]
    x[np.where(np.all([dds1 <= thr, dds1 >= -thr], axis=0))[0]] = symbols[1]
    x[np.where(thr <= dds1)[0]] = symbols[2]
    x[-1] = x[-2]

    return x


def RiseAmp(Signal, t):
    # detect all valleys
    val = detect_peaks(Signal, valley=True)
    # final pks array
    pks = []
    thr = ((np.max(Signal) - np.min(Signal)) * t) + np.mean(Signal)
    # array of amplitude of rising with size of signal
    risingH = np.zeros(len(Signal))
    Rise = np.array([])

    for i in range(0, len(val) - 1):
        # piece of signal between two successive valleys
        wind = Signal[val[i]:val[i + 1]]
        # find peak between two minimums
        pk = detect_peaks(wind, mph=0.1 * max(Signal))
        # print(pk)
        # if peak is found:
        if (len(pk) > 0):
            # append peak position
            pks.append(val[i] + pk)
            # calculate rising amplitude
            risingH[val[i]:val[i + 1]] = [wind[pk] - Signal[val[i]] for a in range(val[i], val[i + 1])]
            Rise = np.append(Rise, (wind[pk] - Signal[val[i]]) > thr)

    risingH = np.array(risingH > thr).astype(int)
    Rise = Rise.astype(int)

    return risingH


# Auxiliary methods
def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):
    """Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb"""

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan - 1, indnan + 1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                       & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind


def _plot(x, mph, mpd, threshold, edge, valley, ax, ind):
    """Plot results of the detect_peaks function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(x, 'b', lw=1)
        if ind.size:
            label = 'valley' if valley else 'peak'
            label = label + 's' if ind.size > 1 else label
            ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                    label='%d %s' % (ind.size, label))
            ax.legend(loc='best', framealpha=.5, numpoints=1)
        ax.set_xlim(-.02 * x.size, x.size * 1.02 - 1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)
        ax.set_xlabel('Data #', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        mode = 'Valley detection' if valley else 'Peak detection'
        ax.set_title("%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"
                     % (mode, str(mph), mpd, str(threshold), edge))


def merge_chars(string_matrix):
    """
    Function performs the merge of the strings generated with each method. The function assumes
    that each string is organized in the StringMatrix argument as a column.
    The function returns the merged string.
    """
    col = np.size(string_matrix, axis=0)
    lines = np.size(string_matrix, axis=1)
    Str = ""
    for l in range(0, lines):
        for c in range(0, col):
            Str += str(string_matrix[c][l])

    return Str


def vmagnitude(v):
    """
    Returns the magnitude of a tridimensional vector signal.
    :param v: (ndarray-like)

    :return: The magnitude of the signal.
    """
    return np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2 + v[:, 2] ** 2)


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def prep_str(cfgstr, ls):
    """This function prepares the """
    if r' | ' in cfgstr:
        pstr = cfgstr.split(r' | ')
    else:
        pstr = [cfgstr] * ls
    return pstr


def plot_matches(s, m, scatter=False):
    figure(figsize=(16, 5))
    plot(s)
    if (scatter):
        [plot(i[0], s[i[0]], 'o', color='r') for i in m];
    else:
        [vlines(i[0], np.min(s), np.max(s), lw=3) for i in m];

def pre_processing(s, processing_methods):
    s = np.asarray(s)
    if s.ndim == 1:
        s = np.array([s])

    ns2 = np.zeros(np.shape(s))

    pp_str = prep_str(processing_methods, len(ns2))

    operands = ""

    for i in range(len(s)):
        pp_func_stack = pp_str[i].split(" ")
        ns2[i, :] = s[i]
        for j, val in enumerate(pp_func_stack):
            print(val)
            if not isfloat(val):
                if val is "":
                    print("Space")
                    continue
                elif val not in list(gots_func_dict["pre_processing"].keys()):
                    sys.exit('Unknown pre-processing symbol.')
                else:
                    operator = val
                    # print(gots_func_dict["pre_processing"][operator])
                    for subval in pp_func_stack[j + 1:]:
                        if not isfloat(subval):
                            break
                        else:
                            operands += subval + ','

                    if operands is "":
                        ns2[i, :] = eval(gots_func_dict["pre_processing"][operator] + '(ns2[' + str(i) + ',:])')

                    else:
                        ns2[i, :] = eval(
                            gots_func_dict["pre_processing"][operator] + '(ns2[' + str(i) + ',:],' + operands[:-1] + ')')
                    operands = ""
            else:
                continue
        if(len(s)==1):
            return ns2[0]
        else:
            return ns2

def connotation(s, connotation):
    sc_str = prep_str(connotation, len(s))
    operands = ""
    merged_sc_str = []
    for i in range(len(s)):
        sc_func_stack = sc_str[i].split(" ")
        for j, val in enumerate(sc_func_stack):
            if not isfloat(val):
                if val is "":
                    sys.exit('At least a connotation method must be supplied.')
                elif val[0] == "[" and val[-1] == "]":
                    continue
                elif val not in list(gots_func_dict["connotation"].keys()):
                    sys.exit('Unknown connotation symbol.')
                else:
                    operator = val
                    for subval in sc_func_stack[j + 1:]:
                        if not isfloat(subval) and subval[0] != "[" and subval[-1] != "]":
                            break
                        else:
                            operands += subval + ','

                    if operands is "":
                        _constr = eval(gots_func_dict["connotation"][operator] + '(s[' + str(i) + '])')

                    else:
                        _constr = eval(
                            gots_func_dict["connotation"][operator] + '(s[' + str(i) + '],' + operands[:-1] + ')')

                    operands = ""
                    merged_sc_str += [_constr]
            else:
                continue
    constr = merge_chars(merged_sc_str)

    return constr, merged_sc_str

def symbolic_search(constr, merged_sc_str, search):
    matches = []
    regit = re.finditer(search, constr)
    [matches.append((int(i.span()[0] / np.shape(merged_sc_str)[0]),
                     int(i.span()[1] / np.shape(merged_sc_str)[0]))) for i in regit]
    return matches

# Main methods
def ssts(s, cfg, report='clean'):
    """
    Performs a query on a given time series based upon on a syntactic approach.
    :param s: array-like
        The input time series.
    :param cfg: dictionary
        A configuration dictionary structure that defines the query parameters.
    :param report: string
        A string containing the report type of returned values. Available types
        include: ``clean``, which returns the positive matches only and ``full``
        which returns the pre-processed signal, the connotation string and the
        positive matches.
    :return:
        ns: (array-like)
        The signal segment that corresponds to the query result.
    """
    # Handles exception to multisignal approach.

    ns = np.copy(s)

    # Layer 1: Pre-processing
    # print(cfg["pre_processing"])
    ns = pre_processing(ns, cfg["pre_processing"])

    #Layer 2: Connotation
    constr, merged_sc_str = connotation(ns, cfg["connotation"])

    #Layer 3: Search
    matches = symbolic_search(constr, merged_sc_str, cfg["expression"])

    # removes unnecessary nesting in case ndim is 1.
    if ns.shape[0] == 1:
        ns = ns[0]

    if report is 'clean':
        return matches
    elif report is 'full':
        return ns, constr, matches

def CharFreqTest(s_string1, s_string2, n_connotations, win):
    freq = {}
    if (n_connotations == "derivative"):
        s_string_m1 = s_string1[:win // 2][::-1] + s_string1 + s_string1[-win // 2:][::-1]
        s_string_m2 = s_string2[:win // 2][::-1] + s_string2 + s_string2[-win // 2:][::-1]

        for char in set(s_string_m1):
            freq[char] = []

        for i in range(0, len(s_string_m1)-win):
            string_i1 = s_string_m1[i:i+win]
            string_i2 = s_string_m2[i:i+win]

            freq_i, names_i = string_corr4char_count(string_i1, string_i2, ["p", "z", "n"])

            for ii in freq_i:
                freq[ii].append(freq_i[ii])
        return freq


def CharFreq(s_string, n_connotations, win):
    freq = {}
    time = {}

    if (n_connotations == 1):
        #mirror string
        s_string_m = s_string[:win//2][::-1] + s_string + s_string[-win//2:][::-1]
        for char in set(s_string_m):
            freq[char] = []
            time[char] = []
        for i in range(0, len(s_string_m)-win):
            string_i = s_string_m[i:i+win]
            cnt_data = Counter(string_i)
            for ii in set(s_string):
                if (ii not in list(cnt_data)):
                    freq[ii].append(0)
                else:
                    freq[ii].append(cnt_data[ii] / len(string_i))
        return freq

def string_corr4char_count(s1, s2, seq):
    """
    Gives a distribution of the different characters of string 1 that appear related with a character in string 2
    :param s1: string 1
    :param s2: string 2
    :return: distribution of frequency of the set of characters for both strings
    """
    str_list1 = list(s1)
    str_list2 = list(s2)
    freq = {}
    for char_1 in seq:
        freq[char_1] = []

        for char_2 in seq:
            freq_i = [1 for a_i, b_i in zip(str_list1, str_list2) if(a_i==char_1 and b_i==char_2)]
            freq[char_1].append(len(freq_i))


    return freq, list(set(s2))

def string_corr(s1, s2, ):
    """
    Gives a distribution of the different characters of string 1 that appear related with a character in string 2
    :param s1: string 1
    :param s2: string 2
    :return: distribution of frequency of the set of characters for both strings
    """
    str_list1 = list(s1)
    str_list2 = list(s2)
    freq = {}
    for char_1 in set(s1):
        freq[char_1] = []

        for char_2 in list(set(s2)):
            freq_i = [1 for a_i, b_i in zip(str_list1, str_list2) if(a_i==char_1 and b_i==char_2)]
            freq[char_1].append(len(freq_i))


    return freq, list(set(s2))


def string_corr_trans(s1, s2):
    """
    Gives a distribution of the different characters of string 1 that appear related with a character in string 2
    :param s1: string 1
    :param s2: string 2
    :return: distribution of frequency of the set of characters for both strings
    """
    str_list1 = list(s1)
    str_list2 = [s2[s2_i:s2_i+2] for s2_i in range(0, len(s2)-1)]
    set_list2 = [i for i in list(set(str_list2)) if(i[0]!=i[1])]

    freq = {}
    for char_1 in set(s1):
        freq[char_1] = []
        for char_2 in set_list2:
            freq_i = [1 for a_i, b_i in zip(str_list1, str_list2) if(a_i==char_1 and b_i==char_2)]
            freq[char_1].append(len(freq_i))


    return freq, set_list2

def Output1(s_string, matches, n_connotations, temp_array="False"):
    freq = {}
    time = {}
    if (n_connotations == 1):
        for char in set(s_string):
            freq[char] = []
            time[char] = []
        for i in matches:
            a = s_string[i[0]:i[1]]
            cnt_data = Counter(a)

            for ii in set(s_string):

                if (ii not in list(cnt_data)):
                    freq[ii].append(0)
                else:
                    freq[ii].append(cnt_data[ii] / len(a))

                if (temp_array is "False"):
                    time[ii].append((i[1] - i[0]) / 2)
                else:
                    time[ii].append((temp_array[i[0]] + (temp_array[i[1]] - temp_array[i[0]]) / 2))

    return time, freq

def levenshteinDist(s1, s2):
    d = np.zeros((len(s1), len(s2)))

    for i in range(0, len(s1)):
        d[i, 0] = i
    for j in range(0, len(s2)):
        d[0, j] = j

    for j in range(0, len(s2)):
        for i in range(0, len(s1)):
            if (s1[i] == s2[j]):
                cost = 0
            else:
                cost = 1
            d[i, j] = min([d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost])

    return d[-1, -1]


def string_matrix(string, matches, method="levenshtein"):
    d = np.zeros((len(matches), len(matches)))

    if (method == "levenshtein"):
        for i, match1 in enumerate(matches):
            s1 = string[match1[0]:match1[1]]
            for j, match2 in enumerate(matches):
                s2 = string[match2[0]:match2[1]]
                d[i, j] = levenshteinDist(s1, s2)

    return d

def convolve1d(s1, s2):
    l_s1 = np.size(s1)
    l_s2 = np.size(s2)

    C = np.zeros(l_s1 + l_s2 - 1)

    for m in range(l_s1):
        for n in range(l_s2):
            C[m + n] = C[m + n] + (s1[m] * s2[n])
    return C / (l_s1 + l_s2 - 1)


def cross_corr(s1, s2):
    C = convolve1d(np.conj(s1), s2)

    return C


def kind_of_Similarity(a, b):
    if (len(a) > len(b)):
        major = a
        minor = b
    elif (len(b) > len(a)):
        major = b
        minor = a
    else:
        major = a
        minor = b
    major = major[::-1]
    ab_sub = []
    for i in range(len(b) + len(a)):
        win = np.zeros(len(minor))
        if (i <= len(minor)):
            win[0:i] = major[0:i][::-1]
        elif (i > len(minor) and i <= len(major)):
            win = major[i - len(minor):i][::-1]
        else:
            win[i - len(major):] = major[i - len(minor):][::-1]

        ab_sub.append(np.sum(abs(np.subtract(minor, win))))

    return sum(ab_sub) / (len(major) + len(minor))


def numerical_matrix(data_array, matches, temp_array, method):
    d = np.zeros((len(matches), len(matches)))

    if (method == "convolve"):
        for i, match1 in enumerate(matches):
            s1 = data_array[match1[0]:match1[1]]
            for j, match2 in enumerate(matches):
                s2 = data_array[match2[0]:match2[1]]
                # plt.figure()
                # ax1 = plt.subplot(3,1,1)
                # ax2 = plt.subplot(3,1,2)
                # ax3 = plt.subplot(3,1,3)
                s1_1 = s1 - np.mean(s1)
                s2_2 = s2 - np.mean(s2)
                # ax1.plot(s1_1)
                # ax2.plot(s2_2)
                # ax3.plot(cross_corr(s1_1, s2_2))
                # plt.show()
                # x = cross_corr(s1_1, s2_2)
                # stat, pvalue = ttest_ind(s1_1, s2_2)
                val = kind_of_Similarity(s1_1, s2_2)
                print(val)
                d[i, j] = val
        return d

    elif (method == "slope"):
        slope_m = []
        slope_t = []
        for i, match1 in enumerate(matches):
            slice_data = data_array[match1[0]:match1[1]]

            if (len(slice_data) == 1):
                slope_m.append(0)
            else:
                slope_m.append(np.mean(np.diff(slice_data)))
            slope_t.append((temp_array[match1[0]] + (temp_array[match1[1]] - temp_array[match1[0]]) / 2))
        return slope_t, slope_m

    elif( method =="dtw"):
        for i, match1 in enumerate(matches):
            s1 = data_array[match1[0]:match1[1]]
            s1_1 = s1 - np.mean(s1)
            for j, match2 in enumerate(matches):
                s2 = data_array[match2[0]:match2[1]]
                s2_2 = s2 - np.mean(s2)
                val = dtw_distance(s1_1, s2_2)

                d[i, j] = val
        return d

    elif (method == "time_width"):
        width = []
        width_t = []
        for i, match1 in enumerate(matches):
            width.append(to_integer(temp_array[match1[1]] - temp_array[match1[0]]))
            width_t.append((temp_array[match1[0]] + (temp_array[match1[1]] - temp_array[match1[0]]) / 2))
        return width_t, width


def to_integer(dt_time):
    return dt_time.days

def dtw_distance(a,b):
    euclidean_norm = lambda a, b: np.abs(a - b)

    d, cost_matrix, acc_cost_matrix, path = dtw(a, b, dist=euclidean_norm)

    print(d)

    return d

def probability_Char(data):
    keys = list(data.keys())
    total_freq = 0
    for key in keys:
        total_freq += data[key]
    pb = {}
    for i in range(0, len(data[keys[0]])):
        for key in keys:
            pb[key] = data[key][i]/data