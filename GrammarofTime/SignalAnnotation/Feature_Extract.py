import tsfel
from tools.load_tools import loadH5, load_npz_featuresHui
from tools.plot_tools import plotFeaturesTSFLBased, plotScatterColors, plotLabelsColors, plot_textcolorized
from tools.processing_tools import featuresTsfel, featuresTsfelMat, WindowStat, chunk_data
from tools.string_processing_tools import runLengthEncoding, Ngrams, NgramsInt, NgramsPos
from GrammarofTime.SSTS.backend.gotstools import *
from PDF_generator.reportGen import Report
from definitions import CONFIG_PATH
from pandas import read_json
import time
from multiprocessing import Pool

#clustering
from sklearn.cluster import DBSCAN, KMeans, AffinityPropagation, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import string

def multiclusterConversion(labels, n=2):
    alphabet = string.ascii_lowercase

    str_labels = np.array([("".join(str(a) for a in label)) for label in labels.T])

    code = {key: alphabet[i] for i, key in enumerate(set(str_labels))}
    str_decoded = [code[code_i] for code_i in str_labels]

    #rle
    decoded_join, rle_decoded, rep_decoded = runLengthEncoding(str_decoded)
    decoded_wordLst, ngrams_decoded = Ngrams(rle_decoded, n=n)
    pos_decoded = NgramsPos(rep_decoded,n=n)


    return ngrams_decoded, pos_decoded

def plot_posSet(signal, prob_sig, labels):

    ngrams_decoded, pos_decoded = multiclusterConversion(labels, n=2)
    # print(ngrams_decoded)
    # print(set(ngrams_decoded))
    transitions = {}
    for transition in set(ngrams_decoded):
        # print(transition)
        transitions[transition] = []
        indx = np.where(np.array(ngrams_decoded)==transition)[0]
        # print(indx)
        for indx_i in indx:
            transitions[transition].append(np.sum(prob_sig[pos_decoded[indx_i][0]:pos_decoded[indx_i][1]]))


def normalization(s):
    if(np.max(abs(s))==0):
        return s
    else:
        return s / np.max(abs(s))

def loadfeaturesbydomain_sub(features, featureSet):
    for feature in features["features"].keys():
        # print(np.where(np.isnan(features["features"][feature])))
        if (feature in ["stat_m_abs_dev", "spec_m_coeff", "temp_mslope", "spec_f_f"]):
            continue
        elif(len(np.where(np.isnan(features["features"][feature]))[0])>0):
            continue
        else:
            # print(feature)
            # print(len(features["features"][feature]))
            signal_i = features["features"][feature]
            signal_i = normalization(signal_i)
            featureSet['allfeatures'].append(signal_i)
            if ("temp" in feature):
                featureSet['featurebydomain']["temp"].append(signal_i)
            elif ("spec" in feature):
                featureSet['featurebydomain']["spec"].append(signal_i)
            else:
                featureSet['featurebydomain']["stat"].append(signal_i)
    return featureSet


def load_featuresbydomain(file, features_tag="all"):
    featureSet = dict(featurebydomain={"temp": [], "stat": [], "spec": []}, allfeatures=[])
    if(features_tag=="all"):
        for features in file:
            featureSet = loadfeaturesbydomain_sub(features, featureSet)
    else:

        featureSet = loadfeaturesbydomain_sub(file[features_tag], featureSet)

    return featureSet

def getLabels(X, n_clusters):
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward", affinity="euclidean").fit(X)
    labels = agg.labels_
    return labels

def signalClustering(file, tag, n_clusters, lims):

    feature_set = load_featuresbydomain(file, features_tag=tag)
    X = np.array(feature_set["allfeatures"]).T[lims[0]:lims[1], :]
    X_pca = PCAclustering(X)
    X = StandardScaler().fit_transform(X_pca)
    return getLabels(X, n_clusters)

def checkAnomalyChange(labels, win):
    s = np.zeros(np.shape(labels))
    for i, label in enumerate(labels):
        divider = np.sum(np.repeat([chunk_data(label, win//2)], win//2, axis=1), axis=2)[0][:len(label)]
        s[i, :] = label/np.where(divider== 0, 1, divider)

    return s

def rangeBetweenIndxs(pdensity, labels):
    alphabet = string.ascii_lowercase
    peaks_ind = detect_peaks(pdensity)
    categorize_labels = []
    str_labels = [("".join(str(a) for a in label)) for label in labels.T]
    code = {key: alphabet[i] for i, key in enumerate(set(str_labels))}

    str_decoded = [code[code_i] for code_i in str_labels]
    cnt = 0
    for i in range(len(peaks_ind)):
        if(i==0):
            print(peaks_ind)
            str_temp = str_decoded[:peaks_ind[i]]
            length = peaks_ind[i]
            categorize_labels += length*[max(set(str_temp), key=str_temp.count)]
        elif(i==len(peaks_ind)-1):

            str_temp1 = str_decoded[peaks_ind[i-1]:peaks_ind[i]]
            str_temp2 = str_decoded[peaks_ind[i]:]
            length1 = peaks_ind[i] - peaks_ind[i-1]
            length2 = len(str_labels)-peaks_ind[i]
            categorize_labels += length1*[max(set(str_temp1), key=str_temp1.count)]
            categorize_labels += length2*[max(set(str_temp2), key=str_temp2.count)]
        else:
            str_temp = str_decoded[peaks_ind[i-1]:peaks_ind[i]]
            length = peaks_ind[i] - peaks_ind[i-1]
            categorize_labels+= length*[max(set(str_temp), key=str_temp.count)]

        cnt+=length
    print(cnt)

    ax = plt.subplot(1,1,1)
    print(len(str_labels))
    print(len(categorize_labels))
    print(len(pdensity))
    plot_textcolorized(pdensity, categorize_labels[:-1], ax)
    plt.show()



def pDensityofChangebyDomain(labels):
    changes = np.copy(labels)
    dchanges = np.where(np.diff(changes)!=0, 1, 0)
    dchanges = checkAnomalyChange(dchanges, 50)
    dchanges = 5000*np.sum(dchanges, axis=0)
    s = np.ones(np.shape(dchanges))
    win = 50
    ch = np.r_[dchanges[win//2:0:-1], dchanges, dchanges[-1:-win//2:-1]]

    for i in range(win//2, len(s)+win//2):
        # print(i)
        # print(np.sum(ch[i-win//2:i+win//2]))
        s[i-win//2] = np.sum(ch[i-win//2:i+win//2])

    s = smooth(s, window_len=win*2)
    # plt.plot(s)
    # plt.show()
    # changes = np.zeros(np.shape(labels))
    # for i, label in enumerate(labels):
    #     change = np.diff(label)
    #     change = np.append(change, label[0])
    #     changes[i,:] = np.where(np.diff(labels)!=0, 1, 0)
    #
    # change = 10*np.sum(changes, axis=1)

    return s

def PCAclustering(X):
    pca_model = PCA(n_components=0.95)
    pca_fit = pca_model.fit(X)
    X_pca = pca_fit.transform(X)

    return X_pca

def signalClusteringbyDomain(file, tag, n_clusters, lims):

    feature_set = load_featuresbydomain(file, features_tag=tag)


    # print(pca_fit.explained_variance_ratio_)
    # print(np.cumsum(pca_fit.explained_variance_ratio_))

    X_spec = np.array(feature_set["featurebydomain"]["spec"]).T[lims[0]:lims[1],:]
    X_pca_spec = PCAclustering(X_spec)
    X_spec = StandardScaler().fit_transform(X_pca_spec)

    X_stat = np.array(feature_set["featurebydomain"]["stat"]).T[lims[0]:lims[1],:]
    X_pca_stat = PCAclustering(X_stat)
    X_stat = StandardScaler().fit_transform(X_pca_stat)

    X_temp = np.array(feature_set["featurebydomain"]["temp"]).T[lims[0]:lims[1],:]
    X_pca_temp = PCAclustering(X_temp)
    X_temp = StandardScaler().fit_transform(X_pca_temp)

    spec_labels = getLabels(X_spec, n_clusters)
    stat_labels = getLabels(X_stat, n_clusters)
    temp_labels = getLabels(X_temp, n_clusters)

    titles = ["Temp Features", "Stat Features", "Spec Features"]

    return np.array([temp_labels, stat_labels, spec_labels]), titles

def plotsignalClusteringbyDomain(signal, file, tag, n_clusters, suptitle):
    a = 35000
    b = 45000
    labels, titles = signalClusteringbyDomain(file, tag, n_clusters, lims=(a,b))
    pdensity = pDensityofChangebyDomain(labels)
    figs, axs = plt.subplots(len(labels), 1, sharex="all")
    plt.suptitle(suptitle)

    for (i, label), title in zip(enumerate(labels), titles):
        axs[i].plot(pdensity)
        plotScatterColors(signal[a:b, tag], ref_signal[a:b], label, title, axs[i])

    plt.show()

    plot_posSet(signal[a:b, tag], pdensity, labels)


def multiSignalClustering(file, tags, n_clusters):
    labels_array = np.array([])
    a = 75000
    b = 90000
    for tag in tags:
        feature_set = load_featuresbydomain(file, features_tag=tag)
        X = np.array(feature_set["allfeatures"]).T[a:b, :]
        X = StandardScaler().fit_transform(X)
        labels_array = np.append(labels_array, getLabels(X, n_clusters))

    return labels_array

def plotMultiSignalClustering(signal, file, tags, n_clusters):
    """
    :param signal:array of arrays with the group of original signals
    :param file: file with the set of features associated with each of the signals
    :param tags: tags that represent which signal will be selected for clustering
    :param n_clusters: number of clusters
    :return: plots
    """
    a = 45000
    b = 75000
    ref_signal = signal[:, -1]
    if(len(tags)==1):
        ax = plt.subplot(1,1,1)
        labels = signalClustering(file, tags[0], n_clusters, lims=(a, b))
        plotScatterColors(signal[a:b, tags[0]], ref_signal[a:b], labels, ax)
    else:
        figs, axs = plt.subplots(len(tags), 1, sharex="all")
        for i, tag in enumerate(tags):
            labels = signalClustering(file, tag, n_clusters, lims=(a,b))
            plotScatterColors(signal[a:b, tag], ref_signal[a:b], labels, axs[i])

    plt.show()

guide = CONFIG_PATH + "/Hui_SuperProject/MovDict.json"
guide_dict = read_json(guide).to_dict()
example_path = CONFIG_PATH + "/Hui_SuperProject/Data_Examples/"
key = 15

features_path = CONFIG_PATH + "/GrammarofTime/SignalAnnotation/Features_tsfel_HuiData/"
filename = "features_250_15.npz"
tag = 9
all_signals= loadH5(example_path + guide_dict[key]["file"])
or_signal = all_signals[:,tag:tag+3]
ref_signal = all_signals[:,-1]

file = load_npz_featuresHui(features_path+filename)

tags = [0, 1]

title = guide_dict[key]["phrase"]


a = 45000
b = 65000
# labels = signalClusteringbyDomain(file, tag, 2, lims=(a,b))
# pdensity = pDensityofChangebyDomain(labels)
# rangeBetweenIndxs(pdensity, labels)
# detect_peaks(pdensity, show=True)



# plotsignalClusteringbyDomain(all_signals, file, 0, 4, title)
# plotsignalClusteringbyDomain(all_signals, file, 1, 2, title)
plotsignalClusteringbyDomain(all_signals, file, 8, 4, title)
# plotsignalClusteringbyDomain(all_signals, file, 9, 3, title)
# plotMultiSignalClustering(all_signals, file, [8], 4)

# feature_set1 = load_featuresbydomain(file, features_tag=tag)
# feature_set2 = load_featuresbydomain(file, features_tag=tag+1)
# feature_set3 = load_featuresbydomain(file, features_tag=tag+2)

# X_temp = np.array(feature_set["featurebydomain"]["temp"]).T
a = 30000
b = 40000
# X_spec = np.array(feature_set["featurebydomain"]["spec"]).T
# print(np.where(np.isnan(X_spec)))
# X_stat = np.array(feature_set["featurebydomain"]["stat"]).T
# X_all = np.array(feature_set1["allfeatures"]).T[a:b,:]
# X_all2 = np.array(feature_set2["allfeatures"]).T[a:b,:]
# X_all3 = np.array(feature_set3["allfeatures"]).T[a:b,:]
# X_spec = StandardScaler().fit_transform(X_spec)

# X_spec = StandardScaler().fit_transform(np.transpose(X_spec))
# X_stat = StandardScaler().fit_transform(np.transpose(X_stat))
# X_all = StandardScaler().fit_transform(X_all)
# X_all2 = StandardScaler().fit_transform(X_all2)
# X_all3 = StandardScaler().fit_transform(X_all3)

# cl = 3
# km = AgglomerativeClustering(n_clusters=cl, linkage="ward", affinity="euclidean").fit(X_all)
# km2 = AgglomerativeClustering(n_clusters=cl, linkage="ward", affinity="euclidean").fit(X_all2)
# km3 = AgglomerativeClustering(n_clusters=cl, linkage="ward", affinity="euclidean").fit(X_all3)
# labels = km.labels_
# labels2 = km2.labels_
# labels3 = km3.labels_


# plt.plot(or_signal[a:b])
# plt.plot(labels)
# plt.show()
# fig, axs = plt.subplots(3,1, sharex="all")
# plt.suptitle(guide_dict[key]["phrase"])
# plotScatterColors(or_signal[a:b, 0], ref_signal[a:b], labels, axs[0])
# plotScatterColors(or_signal[a:b, 1], ref_signal[a:b], labels2, axs[1])
# plotScatterColors(or_signal[a:b, 2], ref_signal[a:b], labels3, axs[2])
# plt.show()
#
#
# key = 13
# a = 10000
# b = 30000
# sig = loadH5(example_path + guide_dict[key]["file"])[a:b, :]
# for i, signal in enumerate(file):
#     sig_i = sig[:, i]/np.max(sig[:,i])
#
#     plotFeaturesTSFLBased(sig_i, signal, (a,b))



# keys = [1, 2, 3, 13, 14, 15, 16]
# # for key in guide_dict.keys():
#
# for key in keys:
#     signal = loadH5(example_path + guide_dict[key]["file"])
#     print("key: "+ str(key))
#     print("feature extraction started")
#     t1 = time.time()
#     feat_dict250 = featuresTsfelMat(signal, 1000, window_len=250)
#     file1 = np.savez("D:/PhD/Code/GrammarofTime/SignalAnnotation/Features_tsfel_HuiData/features_250_"+str(key)+".npz", arr=feat_dict250, tag=guide_dict[key]["phrase"])
#     t2 = time.time()
#     print(t2 - t1)
#     feat_dict500 = featuresTsfelMat(signal, 1000, window_len=500)
#     file2 = np.savez("D:/PhD/Code/GrammarofTime/SignalAnnotation/Features_tsfel_HuiData/features_500_"+str(key)+".npz", arr=feat_dict500, tag=guide_dict[key]["phrase"])
#     t3 = time.time()
#     print(t3-t2)
#     feat_dict1000 = featuresTsfelMat(signal, 1000, window_len=1000)
#     file3 = np.savez("D:/PhD/Code/GrammarofTime/SignalAnnotation/Features_tsfel_HuiData/features_1000_"+str(key)+".npz", arr=feat_dict1000, tag=guide_dict[key]["phrase"])
#     t4 = time.time()
#     print(t4-t3)
#     print("feature extraction ended")