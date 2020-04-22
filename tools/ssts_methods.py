from GrammarofTime.SSTS.backend.gotstools import DiffC, AmpC, symbolic_search, connotation, pre_processing
from scipy import optimize
from numpy import piecewise, linspace
from tools.string_processing_tools import runLengthEncoding, Ngrams, tf_idf, tf
from tools.processing_tools import chunk_data_str, mean_norm
from sklearn.feature_extraction import DictVectorizer
import matplotlib.pyplot as plt

import numpy as np

import pwlf

def addArrayofStrings(array_strings):
    return ["".join(list(group_str)) for group_str in np.array(array_strings).T]

def ssts_features():
    """features for the ssts methods:
    1 - len of different chars or len of different ngrams
    """

def piecewise_linear(x, x0, y0, k1, k2):
    return piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])

def PLA(x, y):
    p , e = optimize.curve_fit(piecewise_linear, x, y)
    y_linear = piecewise_linear(x, *p)

    return y_linear

def PLA2(x, y):
    my_pwlf = pwlf.PiecewiseLinFit(x, y)
    print(my_pwlf)

def windowing_str_hy1(str_signal, win_size):
    """In this process, the term frequency in each window is calculated"""

    chunk_str = chunk_data_str(str_signal, window_size=win_size, overlap_size=win_size-1)

    unique, counts = np.unique(chunk_str, return_counts=True, axis=2)
    print(unique)
    print(counts)

    # dict_set = {key:np.sum(np.char.count(chunk_str, key), axis=1) for key in set(str_signal)}
    # mat_set = [np.sum(np.char.count(chunk_str, key), axis=1) for key in set(str_signal)]

    # plt.matshow(mat_set, aspect="auto")

    # plt.show()


def windowing_str(str_signal, win_size):
    chunk_str = chunk_data_str(str_signal, window_size=win_size)
    docs = []
    # Perform RLE
    ind_end = win_size - (len(str_signal) % win_size)
    for i, str_window in enumerate(chunk_str):
        if (i == len(chunk_str) - 1):

            rle_chunk_cnt, seq_str, cnt_str = runLengthEncoding(str_window[:-ind_end])
        else:
            rle_chunk_cnt, seq_str, cnt_str = runLengthEncoding(str_window)

        seq_ngramed = Ngrams(seq_str, 2)[1]
        seq_str = " ".join(str_i for str_i in seq_ngramed)
        docs += [seq_str]

    return seq_str, seq_ngramed, docs

def ssts_ADmethod(signal):
    pp_sig = pre_processing(signal, "S 10")
    constr, merged_sc_str = connotation([pp_sig], "AQ 4 D1 0.05")

    lst_merged_chars = ["".join(row) for row in np.array(merged_sc_str).T]

    return constr, merged_sc_str, lst_merged_chars


def ssts_peakDetector(signal):
    # pla_signal = PLA(linspace(0, len(signal), len(signal)), signal)
    constr, merged_sc_str = connotation([signal], "D1 0.05")

    matches = symbolic_search(constr, merged_sc_str, "p[zn]")

    matches_point = [((match[1]-match[0])//2)+match[0] for match in matches]

    return matches_point

def ssts_segmentDetection(signal):

    constr, merged_sc_str = connotation([signal], "D1 0.05")

    matches = symbolic_search(constr, merged_sc_str, "p+z+n+")

    return matches


def ssts_alignement(sig1, sig2, property="derivative"):
    """try to align two strings from two signals based on a property"""

def ssts_distance(sig1, sig2, property="derivative"):
    """tries to get the distance between 2 different time str representation
        of time series based on a specific property"""

def ssts_changepoint(sig, property):
    """detect change points in the frequency of characters, based on a
        connotation method"""

def ssts_motif(sig, property):
    """find the most relevant motif of a time series sig in character"""

def ssts_warping(sig, property):
    """apply dynamic tw based on a specific property"""

def ssts_distance(sig):
    """Try to design a disrtance method for ssts:
    hypothesis are:
    1 - frequency based distance, whether in
        a - individual chars
        b - ngrams char
        Methods:
            a - Term Frequency
            b - TF-IDF
    2 - string distance based
    """
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    con_str = ssts_ADmethod(sig)

    con_conc = addArrayofStrings(con_str[1])


    # result_join, words, counts = runLengthEncoding(con_str)

    #1 - Distance with individual chars
    # windowing_str_hy1(np.array(con_str[-1]), win_size=100)
    corpus = [" ".join(row) for row in chunk_data_str(np.array(con_conc), window_size=150, overlap_size=149)]
    print(corpus)
    vectorizer = CountVectorizer()
    tfidf = TfidfVectorizer(max_df=0.85, min_df=2, analyzer="word", ngram_range=(20, 20))
    X = vectorizer.fit_transform(corpus)
    Y = tfidf.fit_transform(corpus)
    relevance = np.argsort(np.sum(X.toarray(), axis=0))
    relevanceY = np.argsort(np.sum(Y.toarray(), axis=0))

    X_less_frequent = np.sum(Y.toarray()[:,relevanceY[0:-4]], axis=1)

    plt.matshow(Y.toarray().T, aspect="auto")
    # vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(3, 3))
    # X2 = vectorizer2.fit_transform(corpus)
    # plt.matshow(X2.toarray().T, aspect="auto")
    plt.figure()
    plt.plot(mean_norm(sig))
    plt.plot(mean_norm(X_less_frequent))

    plt.show()










