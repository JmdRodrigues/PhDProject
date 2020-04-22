from scipy import stats
import tsfel
from tools.load_tools import loadH5, load_npz_featuresHui
from tools.plot_tools import plotFeaturesTSFLBased, plotScatterColors, plotLabelsColors, plot_textcolorized
from tools.processing_tools import featuresTsfel, featuresTsfelMat, WindowStat, chunk_data, sumvolve, mean_norm
from tools.string_processing_tools import runLengthEncoding, Ngrams, NgramsInt, NgramsPos, BagofWords
from tools.ssts_methods import ssts_peakDetector, ssts_segmentDetection, ssts_distance
from GrammarofTime.SSTS.backend.gotstools import *
from PDF_generator.reportGen import Report
from definitions import CONFIG_PATH
from pandas import read_json
import time
from multiprocessing import Pool

from suffix_trees import STree

#clustering
from sklearn.cluster import DBSCAN, KMeans, AffinityPropagation, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import string

import os

def ClusterConversion(label, n=2):
    # get alphabet sequence
    alphabet = string.ascii_lowercase

    # get alphabet sequence keys into the clustering sequence (0->a, 1->b...)
    str_labels = np.array([str(a) for a in label])


    code = {key: alphabet[i] for i, key in enumerate(set(str_labels))}
    str_decoded = [code[code_i] for code_i in str_labels]
    print(str_decoded)
    # rle
    decoded_join, rle_decoded, rep_decoded = runLengthEncoding(str_decoded)
    print(rle_decoded)
    decoded_wordLst, ngrams_decoded = Ngrams(rle_decoded, n=n+1)
    pos_decoded = NgramsPos(rep_decoded, n=n+1)
    print(pos_decoded)

    print(ngrams_decoded)
    # bow
    bow = BagofWords(ngrams_decoded)
    new_bow = {k: bow[k] for k in bow if len(set(k)) == n}
    print(new_bow)
    mre_item = max(new_bow, key=lambda k: new_bow[k])
    print(mre_item)

    ngram_indexs = np.where(np.array(ngrams_decoded)==mre_item)[0]
    all_ngram_indexs = [np.where(np.array(ngrams_decoded)==item_i)[0] for item_i in new_bow]


    indexs = [pos_decoded[i][d] for d in range(3) for i in ngram_indexs]
    ngrams_vspan_indexs = [[(pos_decoded[i][0], pos_decoded[i][-1]) for i in indexs_i] for indexs_i in all_ngram_indexs]
    # suffix_tree
    # st = STree.STree(ngrams_decoded)
    # indexs = st.find_all(mre_item)

    return ngrams_decoded, pos_decoded, indexs, ngrams_vspan_indexs

def multiclusterConversion(labels, n=2):
    #get alphabet sequence
    alphabet = string.ascii_lowercase

    #get alphabet sequence keys into the clustering sequence (0->a, 1->b...)
    str_labels = np.array([("".join(str(a) for a in label)) for label in labels.T])

    code = {key: alphabet[i] for i, key in enumerate(set(str_labels))}
    str_decoded = [code[code_i] for code_i in str_labels]

    #rle
    decoded_join, rle_decoded, rep_decoded = runLengthEncoding(str_decoded)
    decoded_wordLst, ngrams_decoded = Ngrams(rle_decoded, n=n)
    pos_decoded = NgramsPos(rep_decoded,n=n)

    #bow
    bow = BagofWords(ngrams_decoded)
    mre_item = max(bow, key=lambda k: bow[k])
    print(mre_item)

    #suffix_tree
    st = STree.STree(ngrams_decoded)
    indexs = st.find_all(mre_item)

    return ngrams_decoded, pos_decoded, indexs

def plot_posSet(signal, prob_sig, labels):

    ngrams_decoded, pos_decoded, indexs = multiclusterConversion(labels, n=2)

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
    dchanges = checkAnomalyChange(dchanges, 20)
    dchanges = np.sum(dchanges, axis=0)
    s = np.ones(np.shape(dchanges))
    win = 20
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

def signalClusteringbyDomain(file, tag, n_clusters, lims, sub, subsampling=True):
    feature_set = load_featuresbydomain(file, features_tag=tag)

    if(subsampling):
        sub = sub
        X_spec = np.array(feature_set["featurebydomain"]["spec"]).T[lims[0]:lims[1]:sub, :]
        X_pca_spec = PCAclustering(X_spec)
        X_spec = StandardScaler().fit_transform(X_pca_spec)

        X_stat = np.array(feature_set["featurebydomain"]["stat"]).T[lims[0]:lims[1]:sub, :]
        X_pca_stat = PCAclustering(X_stat)
        X_stat = StandardScaler().fit_transform(X_pca_stat)

        X_temp = np.array(feature_set["featurebydomain"]["temp"]).T[lims[0]:lims[1]:sub, :]
        X_pca_temp = PCAclustering(X_temp)
        X_temp = StandardScaler().fit_transform(X_pca_temp)
    else:

        X_spec = np.array(feature_set["featurebydomain"]["spec"]).T[lims[0]:lims[1], :]
        X_pca_spec = PCAclustering(X_spec)
        X_spec = StandardScaler().fit_transform(X_pca_spec)

        X_stat = np.array(feature_set["featurebydomain"]["stat"]).T[lims[0]:lims[1], :]
        X_pca_stat = PCAclustering(X_stat)
        X_stat = StandardScaler().fit_transform(X_pca_stat)

        X_temp = np.array(feature_set["featurebydomain"]["temp"]).T[lims[0]:lims[1], :]
        X_pca_temp = PCAclustering(X_temp)
        X_temp = StandardScaler().fit_transform(X_pca_temp)

    # print(pca_fit.explained_variance_ratio_)
    # print(np.cumsum(pca_fit.explained_variance_ratio_))



    spec_labels = getLabels(X_spec, n_clusters)
    stat_labels = getLabels(X_stat, n_clusters)
    temp_labels = getLabels(X_temp, n_clusters)

    titles = ["Temp Features", "Stat Features", "Spec Features"]

    return np.array([temp_labels, stat_labels, spec_labels]), titles

def plotsignalClusteringbyDomain(signal, ref_signal, file, tag, n_clusters, sub, begin_feat, suptitle):
    a = 45000
    b = 85000
    labels, titles = signalClusteringbyDomain(file, tag, n_clusters, lims=(a,b), sub=sub, subsampling=True)
    pdensity = pDensityofChangebyDomain(labels)
    figs, axs = plt.subplots(len(labels)+1, 1, sharex="all")
    plt.suptitle(suptitle)

    colors = ["dodgerblue", "orangered", "lightgreen", "mediumorchid", "gold", "firebrick", "darkorange",
              "springgreen", "lightcoral","dodgerblue", "orangered", "lightgreen", "mediumorchid", "gold", "firebrick", "darkorange",
              "springgreen", "lightcoral"]
    for (i, label), title in zip(enumerate(labels), titles):

        # ngrams_decoded, pos_decoded, indexs, vspan_indexs = ClusterConversion(label, n=n_clusters)
        # print(indexs)
        # print(ref_signal)
        plotScatterColors(signal[:, tag], ref_signal, label, title, axs[i])
        # axs[i].vlines(x=indexs, ymin=0, ymax=max(ref_signal))
        # [[axs[i].axvspan(xmin=indx[0], xmax=indx[1], ymin=0, ymax=max(ref_signal), alpha=0.5, color=colors[c_i]) for indx in indxs_i] for c_i, indxs_i in enumerate(vspan_indexs)]
    axs[-1].plot(pdensity)
    plt.show()

    # print(indexs)

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

def plotImageAndTS(img, signal1, signal2):
    fig, axs= plt.subplots(3,1, sharex=True)

    # im_ax1 = plt.subplot(3, 1, 1)
    # ax2 = plt.subplot(3, 1, 2)
    # ax3 = plt.subplot(3, 1, 3)

    axs[0].matshow(img, aspect="auto")
    axs[1].plot(signal1)
    axs[2].plot(signal2)

    axs[1].set_xlim(0, len(signal1))
    axs[2].set_xlim(0, len(signal1))


def matrix_norm(mat_array):
    # mat_array=mat_array[:, :100]
    print(np.shape(mat_array))
    mat_mean = (mat_array - np.repeat([np.mean(mat_array, axis=1)], np.shape(mat_array)[1], axis=0).T)
    mat_norm = mat_mean / np.repeat([np.std(mat_mean, axis=1)+1], np.shape(mat_array)[1], axis=0).T

    return mat_norm

def combFeaturesPCA(features):
    pca_model = PCA(n_components=0.95)
    X_pca = pca_model.fit_transform(features.T)
    loadings = np.sum(pca_model.components_.T * np.sqrt(pca_model.explained_variance_), axis=0)
    print(np.shape(loadings))
    print(np.shape(X_pca))

    return X_pca, loadings

def K_STest(sig1, sig2):
    _, pval = stats.ks_2samp(sig1, sig2)
    return 1-pval

def ttest(sig1, sig2):
    _, pval = stats.ttest_ind(sig1, sig2)
    return 1 - pval

def DistanceTTest(data, win_size):
    win_change = win_size//2
    output = np.zeros(np.shape(data))
    data_temp = np.r_[data[win_change:0:-1], data, data[-1:len(data) - win_change:-1]]
    for i in range(win_change, len(data)):
        sig1 = data_temp[i-win_change:i]
        sig2 = data_temp[i:i+win_change]

        output[i-win_change] = np.exp(stats.wasserstein_distance(sig1, sig2))

    return output



def ActivitySegments(signal):
    """
    This functions accepts a signal that will be segmented into multiple areas based on the activity.
    The size of each activity signal will not be the same. The process to detect the activity is just to apply the envelope
    of the signal with a really low pass filter. With this wave, peaks of activity can be found and exported. The segments will be the
    area surrounding the maxs, which is by identifying the minimums.
    :param signal: the signal which will be segmented into activity segments
    :return:
    """
    norm_signal = mean_norm(signal)
    activityofS = smooth(abs(norm_signal), window_len=3000)
    matches = ssts_segmentDetection(activityofS)


    plot_matches(norm_signal, matches, "blue", mode="span")
    # plot_matches(activityofS, matches, "blue")
    plt.show()

    return matches



if(os.name=="nt"):
    #running on Windows
    sep = "\\"
else:
    sep="/"

guide = CONFIG_PATH + sep+"Hui_SuperProject"+sep+"MovDict.json"
guide_dict = read_json(guide).to_dict()
example_path = CONFIG_PATH + sep+"Hui_SuperProject"+sep+"Data_Examples"
key = 15

features_path = "D:\PhD\Data\HuisData\Dataset1\FeatureExtracted\Features_tsfel_HuiData"
filename = "features_500_15.npz"
tag = 8
sub = 1
a = 10000
b = 45000
begin_feat = 5000
all_signals = loadH5(example_path + sep+guide_dict[key]["file"])[a:b:sub, :]
or_signal = all_signals[:, tag]
ref_signal = all_signals[:, -1]

ssts_distance(or_signal)

file = load_npz_featuresHui(features_path+sep+filename)

feature_set = load_featuresbydomain(file, features_tag=tag)
features_temp = np.array(feature_set["featurebydomain"]["temp"])[:, a:b:sub]
features_stat = np.array(feature_set["featurebydomain"]["stat"])[:, a:b:sub]
features_spec = np.array(feature_set["featurebydomain"]["spec"])[:, a:b:sub]

features_t_norm = StandardScaler().fit_transform(features_temp)
# # features_t_norm = matrix_norm(features_temp)
# # print(features_t_norm)
features_st_norm = StandardScaler().fit_transform(features_stat)
# # features_st_norm = matrix_norm(features_stat)
features_sp_norm = StandardScaler().fit_transform(features_spec)
# # features_sp_norm = matrix_norm(features_spec)
#
# plotImageAndTS(features_t_norm, np.abs(np.prod(features_t_norm, axis=0)), or_signal)
# plotImageAndTS(features_st_norm, np.abs(np.prod(features_st_norm, axis=0)), or_signal)
# plotImageAndTS(features_sp_norm, np.abs(np.prod(features_sp_norm, axis=0)), or_signal)
# plt.show()
#Combination of features with PCA
# X = np.array(feature_set["allfeatures"]).T


X_pca1, loadings1 = combFeaturesPCA(features_t_norm)
X_pca2, loadings2 = combFeaturesPCA(features_sp_norm)
X_pca3, loadings3 = combFeaturesPCA(features_st_norm)

Xpca1 = X_pca1*loadings1
Xpca2 = X_pca2*loadings2
Xpca3 = X_pca3*loadings3

ax1 = plt.subplot(2,1,1)
ax2 = plt.subplot(2,1,2)
# ax3 = plt.subplot(4,1,3)
# ax4 = plt.subplot(4,1,4)
ax1.plot(or_signal)
ax2.plot(X_pca1[:,0]+10)
ax2.plot(X_pca1[:,1]+5)
ax2.plot(X_pca1[:,2])

plt.show()




wasserDist1 = DistanceTTest(X_pca1[:,0], win_size=125)
wasserDist12 = DistanceTTest(X_pca1[:,1], win_size=125)
wasserDist13 = DistanceTTest(X_pca1[:,2], win_size=125)
wasserDistTotal = wasserDist1+wasserDist12+wasserDist13
# wasserDist2 = DistanceTTest(X_pca1[:,1], win_size=250)
# wasserDist3 = DistanceTTest(X_pca1[:,2], win_size=250)
wasserDist1_2 = DistanceTTest(X_pca2[:,0], win_size=125)+ DistanceTTest(X_pca2[:,1], win_size=125) + DistanceTTest(X_pca2[:,2], win_size=125)
# wasserDist2_2 = DistanceTTest(X_pca2[:,1], win_size=250)
# wasserDist3_2 = DistanceTTest(X_pca2[:,2], win_size=250)
wasserDist1_3 = DistanceTTest(X_pca3[:,0], win_size=125)+ DistanceTTest(X_pca3[:,1], win_size=125) + DistanceTTest(X_pca3[:,2], win_size=125)
# wasserDist2_3 = DistanceTTest(X_pca3[:,1], win_size=250)
# wasserDist3_3 = DistanceTTest(X_pca3[:,2], win_size=250)


# wasserDist_temp = wasserDist1+wasserDist2+wasserDist3
# wasserDist_spect = wasserDist1_2+wasserDist2_2+wasserDist3_2
# wasserDist_stat = wasserDist1_3+wasserDist2_3+wasserDist3_3

#Apply smooth distance

#Detect segments of activity and match with the code segment
matches = ActivitySegments(or_signal)

fig, axs = plt.subplots(4,1, sharex=True)
axs[0].plot(mean_norm(or_signal))
axs[0].plot()
axs[1].plot(smooth(wasserDist1, window_len=250))
axs[2].plot(smooth(wasserDist12, window_len=250))
axs[3].plot(smooth(wasserDist13, window_len=250))


plt.show()
print(len(matches))
for match in matches:
    temp_sig = or_signal[match[0]:match[1]]
    # temp_changepoint1 = smooth(wasserDistTotal[match[0]:match[1]], window_len=250)
    temp_changepoint1= smooth(wasserDist1[match[0]:match[1]], window_len=250)
    temp_changepoint2= smooth(wasserDist12[match[0]:match[1]], window_len=250)
    temp_changepoint3= smooth(wasserDist13[match[0]:match[1]], window_len=250)
    temp_changepointTotal = temp_changepoint1+temp_changepoint2+temp_changepoint3
    ax1 = plt.subplot(2,1,1)
    ax1.plot(temp_sig)
    maxstotal = ssts_peakDetector(temp_changepointTotal)
    # maxs1 = ssts_peakDetector(temp_changepoint1)
    # maxs2 = ssts_peakDetector(temp_changepoint2)
    # maxs3 = ssts_peakDetector(temp_changepoint3)
    maxstotal = np.array(maxstotal)
    # maxs1 = np.array(maxs1)
    # maxs2 = np.array(maxs2)
    # maxs3 = np.array(maxs3)
    ind_maxstotal = np.argsort(temp_changepointTotal[maxstotal])[::-1]
    # ind_maxs1 = np.argsort(temp_changepoint1[maxs1])[::-1]
    # ind_maxs2 = np.argsort(temp_changepoint2[maxs2])[::-1]
    # ind_maxs3 = np.argsort(temp_changepoint3[maxs3])[::-1]
    # # print(maxs[ind_maxs])

    ax1.plot(maxstotal[ind_maxstotal[:4]], temp_sig[maxstotal[ind_maxstotal[:4]]], 'o')
    # ax1.plot(maxs1[ind_maxs1[:5]], temp_sig[maxs1[ind_maxs1[:5]]], 'o')
    # ax1.plot(maxs2[ind_maxs2[:5]], temp_sig[maxs2[ind_maxs2[:5]]], 'o')
    # ax1.plot(maxs3[ind_maxs3[:5]], temp_sig[maxs3[ind_maxs3[:5]]], 'o')
    ax2 = plt.subplot(2,1,2)
    ax2.plot(temp_changepointTotal)
    ax2.plot(maxstotal[ind_maxstotal], temp_changepointTotal[maxstotal[ind_maxstotal]], 'o')
    # ax3 = plt.subplot(4,1,3)
    # ax3.plot(temp_changepoint2)
    # ax3.plot(maxs2[ind_maxs2], temp_changepoint2[maxs2[ind_maxs2]], 'o')
    # ax3 = plt.subplot(4, 1, 4)
    # ax3.plot(temp_changepoint3)
    # ax3.plot(maxs3[ind_maxs3], temp_changepoint3[maxs3[ind_maxs3]], 'o')
    plt.show()




# plt.show()

# plotImageAndTS(features_t_norm, X_pca1 * loadings1, or_signal)
# plotImageAndTS(features_t_norm, X_pca2 * loadings2, or_signal)
# plotImageAndTS(features_t_norm, X_pca3 * loadings3, or_signal)
plt.show()

tags = [0, 1]

title = guide_dict[key]["phrase"]



# labels = signalClusteringbyDomain(file, tag, 2, lims=(a,b))
# pdensity = pDensityofChangebyDomain(labels)
# rangeBetweenIndxs(pdensity, labels)
# detect_peaks(pdensity, show=True)



# plotsignalClusteringbyDomain(all_signals, file, 0, 4, title)
# plotsignalClusteringbyDomain(all_signals, file, 1, 2, title)
# plotsignalClusteringbyDomain(all_signals, ref_signal, file, tag, 4, sub, begin_feat, title)


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