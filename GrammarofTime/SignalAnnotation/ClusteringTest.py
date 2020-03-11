import tsfel

from tools.processing_tools import *
from tools.load_tools import *
import novainstrumentation as ni
from GrammarofTime.SSTS.backend.gotstools import *
from PDF_generator.reportGen import Report
from sklearn.neighbors import KernelDensity
from definitions import CONFIG_PATH
from tools.plot_tools import plot_config, font_config, subplot_pars, plotScatterColors, Csubplot
from pandas import read_json


guide = CONFIG_PATH + "/Hui_SuperProject/MovDict.json"
guide_dict = read_json(guide).to_dict()
example_path = CONFIG_PATH + "/Hui_SuperProject/Data_Examples/"

key = 13

fetures_path = CONFIG_PATH + "/GrammarofTime/SignalAnnotation/Features_tsfel_HuiData/"


signal = loadH5(example_path+guide_dict[key]["file"])[:, 7]
# features250 = np.load(fetures_path+"features_250_"+str(key)+".npz", allow_pickle=True)["arr"].item()
# features500 = np.load(fetures_path+"features_500_"+str(key)+".npz", allow_pickle=True)["arr"].item()
# features1000 = np.load(fetures_path+"features_1000_"+str(key)+".npz", allow_pickle=True)["arr"].item()
# Get features
# Feats = tsfel.time_series_features_extractor(cfg, signal, fs=1000, window_size=100, overlap=1, window_spliter=True)
# print(Feats)
#



features250 = np.load("features_250.npz", allow_pickle=True)["arr"].item()
features500 = np.load("features_500.npz", allow_pickle=True)["arr"].item()
features1000 = np.load("features_1000.npz", allow_pickle=True)["arr"].item()
feature_set = {"250":[], "500":[], "1000":[]}


fig1=plt.figure()
plt.plot(signal/max(signal))
for domain in features250.keys():
    for feat in features250[domain]:
        norm_feat = mean_norm(features250[domain][feat])
        # plt.plot(norm_feat, alpha=0.5, label=feat)
        feature_set["250"].append(features250[domain][feat])

fig2 = plt.figure()
plt.plot(signal / max(signal))
for domain in features500.keys():
    for feat in features500[domain]:
        norm_feat = mean_norm(features500[domain][feat])
        # plt.plot(norm_feat, alpha=0.5, label=feat)
        feature_set["500"].append(features500[domain][feat])

fig3 = plt.figure()
plt.plot(signal / max(signal))
for domain in features1000.keys():
    for feat in features1000[domain]:
        norm_feat = mean_norm(features1000[domain][feat])
        # plt.plot(norm_feat, alpha=0.5, label=feat)
        feature_set["1000"].append(features1000[domain][feat])

plt.legend()
plt.show()


# std_sig = WindowStat(signal, "std", window_len=100, fs=1000)
# zcr_sig = WindowStat(signal, "Azcr", window_len=100, fs=1000)
# sumv_sig = sumvolve(signal, signal[4500:5000])
#
# # plt.plot(tsf_feats["temporal"].values())
# # plt.plot(sumv_sig)
# # plt.plot(signal)
# plt.show()


#clustering
from sklearn.cluster import DBSCAN, KMeans, AffinityPropagation
from sklearn.preprocessing import StandardScaler


X = feature_set["500"]
print(X)
print(np.shape(X))
X = StandardScaler().fit_transform(np.transpose(X))

# aff = AffinityPropagation(damping=0.9, preference=-100).fit(X)

km = KMeans(n_clusters=4).fit(X)
labels = km.labels_

plt.title(guide_dict[key]["phrase"])
plt.plot(signal)
plotScatterColors(signal, labels)
# plt.plot(signal/max(signal))
# plt.plot(std_sig)
# plt.plot(zcr_sig)
plt.plot(labels)
plt.show()
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print(labels)
# print(n_clusters_)
# print(n_noise_)

# plt.plot(signal)
# plt.plot(std_sig)
# plt.show()

#use tsfel

# DEFAULT PARAM for testing
# time_unit = 1e9  # seconds
# resample_rate = 30  # resample sampling frequency
# window_size = 100  # number of points
# overlap = 0  # varies between 0 and 1
#
# # Signal processing
# data_new = tsfel.merge_time_series({"Acc":signal}, resample_rate, time_unit)
# windows = tsfel.signal_window_spliter(data_new, window_size, overlap)
#
# FEATURES_JSON = tsfel.__path__[0] + '/feature_extraction/features.json'
# settings0 = tsfel.load_json(FEATURES_JSON)
# settings1 = tsfel.get_features_by_domain('statistical')
# settings2 = tsfel.get_features_by_domain('temporal')
# settings3 = tsfel.get_features_by_domain('spectral')
#
# features1 = tsfel.time_series_features_extractor(data = tsfel.dataset_features_extractor("D:\PhD\Code\GrammarofTime\SignalAnnotation", settings1, search_criteria=[signal], time_unit=time_unit,
#                                         resample_rate=resample_rate, window_size=window_size, overlap=overlap, output_directory="D:\PhD\Code\GrammarofTime\SignalAnnotation"))
#
# print(features1)
