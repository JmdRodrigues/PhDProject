from scipy.io import loadmat
from tools.plot_tools import Cplot
from tools.processing_tools import *
from tools.string_processing_tools import *
from tools.plot_tools import plot_textcolorized, strsignal2color, plotScatterColors
from tools.load_tools import loadH5
from novainstrumentation import bandpass
from scipy import fftpack
import numpy as np
import matplotlib.pyplot as plt
from GrammarofTime.SSTS.sandbox.connotation_sandbox import AmplitudeTrans, AmpChange, D1Speed, SignConnotation, \
    addArrayofStrings
import string
import time
from novainstrumentation import smooth

from gensim import models, corpora
import pyLDAvis
import pyLDAvis.gensim

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pandas as pd


def Connotation2(sig):
    # amp_level = AmplitudeTrans(sig, 2, string.ascii_uppercase, method="quantiles")

    t0 = time.time()
    ampdiff_str = AmpChange(sig, 0.75, "absolute")

    # ax1 = plt.subplot(1, 1, 1)
    # plot_textcolorized(wave3, wave3_conc_str_tpl[2], ax1)
    # plot_textcolorized(sig, ampdiff_str, ax1)
    # plt.show()
    t1 = time.time()

    print("Done with ampdiff...")
    print("time: " + str(t1 - t0))
    speed_str = D1Speed(sig, 0.75)
    t2 = time.time()
    print("Done with diff...")
    print("time: " + str(t2 - t1))
    sign_str = SignConnotation(sig)
    t3 = time.time()
    print("Done with sign...")
    print("time: " + str(t3 - t2))
    print("creating string...")
    # wave_str = addArrayofStrings([sign_str, ampdiff_str, speed_str])
    wave_str = addArrayofStrings([ampdiff_str, speed_str])

    print("Done")

    return wave_str

def movingLSA(str_signal, win_size=250):

    chunk_str = chunk_data_str(np.array(str_signal), window_size=win_size)
    docs = []
    # Perform RLE
    ind_end = win_size - (len(str_signal) % win_size)
    for i, str_window in enumerate(chunk_str):
        if(i == len(chunk_str)-1):

            rle_chunk_cnt, seq_str, cnt_str = runLengthEncoding(str_window[:-ind_end])
        else:
            rle_chunk_cnt, seq_str, cnt_str = runLengthEncoding(str_window)

        docs.append(seq_str)

    dictionary_LDA = corpora.Dictionary(docs)
    # dictionary_LDA.filter_extremes(no_below=10, no_above=0.5)
    corpus = [dictionary_LDA.doc2bow(list_of_tokens) for list_of_tokens in docs]

    svd = TruncatedSVD(n_components=3)
    lsa = svd.fit_transform(corpus)

    print(list(lsa))

    # topic_extended_df = pd.DataFrame(lsa, columns=["Topic 1", "Topic 2", "Topic 3"])
    # topic_encoded_df["sentence"] = body


# def
def get_key(item):
    return item[1]

def movingLDA(str_signal, win_size=500):
    """
    Scan the str signal in order to extract frequency information
    :param str_signal:
    :param win_size:
    :return:
    """

    # str_signal = np.r_[str_signal[win_size//2:0:-1], str_signal, str_signal[-1:len(str_signal) - win_size//2:-1]].transpose()
    chunk_str = chunk_data_str(np.array(str_signal), window_size=win_size)
    print(len(chunk_str))
    docs = []
    # Perform RLE
    ind_end = win_size - (len(str_signal) % win_size)
    for i, str_window in enumerate(chunk_str):
        if(i == len(chunk_str)-1):

            rle_chunk_cnt, seq_str, cnt_str = runLengthEncoding(str_window[:-ind_end])
        else:
            rle_chunk_cnt, seq_str, cnt_str = runLengthEncoding(str_window)

        seq_ngramed = Ngrams(seq_str, 5)
        # print(seq_ngramed[1])


        docs.append(seq_ngramed[1])

    dictionary_LDA = corpora.Dictionary(docs)
    # dictionary_LDA.filter_extremes(no_below=10, no_above=0.5)
    corpus = [dictionary_LDA.doc2bow(list_of_tokens) for list_of_tokens in docs]

    num_topics = 4

    lda_model = models.LdaMulticore(corpus, num_topics=num_topics,
                                id2word=dictionary_LDA,
                                passes=100, alpha=[0.01] * num_topics,
                                eta=[0.01] * len(dictionary_LDA.keys()))


    vis = pyLDAvis.gensim.prepare(topic_model=lda_model, corpus=corpus, dictionary=dictionary_LDA)
    pyLDAvis.save_html(vis, "test_ldavis.html")
    # pyLDAvis.enable_notebook()
    # pyLDAvis.display(vis)
    # for i, topic in lda_model.show_topics(formatted=True, num_topics=num_topics, num_words=10):
    #     print(str(i) + ": " + topic)
    #     print()


    labels = np.array([]).astype(int)


    for i in range(len(corpus)):
        # labels = np.append(labels, lda_model[corpus[i]][0][0])
        topics = lda_model[corpus[i]]
        topics.sort(key=get_key, reverse=True)
        labels = np.append(labels, np.repeat(topics[0][0], win_size))
        print(topics)
        # print(lda_model[corpus_i])


    return labels[:-ind_end]


example_path = r"/media/jeanraltique/FishStory/Projectos/Doutoramento/PhDCode/PhDProject/Hui_SuperProject/Data_Examples/"

signal = loadH5(example_path + "arthrokinemat_2018_06_03_00_08_52.h5")

fs = 1000
b = 4
acc1 = signal[b * fs:, 9]
acc1_sm = mean_norm(acc1)
acc1_sm = smooth(acc1, 250)

# plt.plot(acc1_sm)
# plt.show()

acc_str = Connotation2(acc1_sm)
# lsa_labels = movingLSA(acc_str)
lda_labels = movingLDA(acc_str)

ax = plt.subplot(1,1,1)
# plot_textcolorized(acc1_sm, acc_str, ax)
plotScatterColors(acc1_sm, signal[b * fs:, -1], lda_labels, title="lda test", ax=ax)
plt.show()
# plt.plot(lda_labels)
# plt.plot(acc1_sm/max(acc1_sm))
# plt.show()
# ax = plt.subplot(1, 1, 1)

# # strsignal2color(acc1_sm, acc_str,ax )
# plt.show()