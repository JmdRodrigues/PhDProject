from __future__ import division

import time
import random as rd
import numpy as np
import matplotlib.pyplot as plt
from novainstrumentation import smooth
from novainstrumentation.peaks import *
from novainstrumentation.smooth import smooth
import novainstrumentation as ni
from tools.PeakFinder import detect_peaks
from scipy.stats import skew, normaltest, kurtosis
import scipy.signal as sc
from tsfel.feature_extraction import features as tsfel_ft
import tsfel
from numpy.lib.stride_tricks import as_strided as ast
import tools.multiprocess_tools as mpt


def quantilestates(sample, quantile_vals):
	alpha = ["a", "b", "c", "d"]
	sample_quantiles = list(quantile_vals)
	sample_quantiles.append(sample)

	return alpha[list(sort(sample_quantiles)).index(sample)]

def quantilstatesArray(signal, quantile_vals, conc=True):
	if(conc):
		return "".join([quantilestates(sample, quantile_vals) for sample in signal])
	else:
		return [quantilestates(sample, quantile_vals) for sample in signal]

def concat_np_strings(mat_string, axis=0):
	"""

	:param mat_string: NxM string matrix which has been generated from a signal
	:param axis:	   concatenate based on the axis selected
	:return: 		   concatenated array
	"""

	return np.apply_along_axis("".join, axis, mat_string)[0]

def prob_hist(x, hist, bins, inverted=False, log=False):

	prob_x = np.zeros(len(x))
	hist_norm = hist/sum(hist)

	for i in range(len(bins) - 1):
		prob_x[np.where(np.logical_and(x > bins[i], x < bins[i + 1]))[0]] = hist_norm[i]
	if(inverted and log):
		return abs(np.log10(1-prob_x))
	elif(inverted and log==False):
		return 1-prob_x
	elif(inverted==False and log):
		return abs(np.log10(prob_x))
	else:
		return prob_x

def hist_dist(x, bin_nbr="fd"):
	hist, bins = np.histogram(x, bins=bin_nbr)

	return hist, bins

def sumvolve(x, window):
	lw=len(window)
	res=np.zeros(len(x)-lw,'d')
	for i in range(len(x)-lw):
		res[i]=sum(abs(x[i:i+lw]-window))/float(lw)
		#res[i]=sum(abs(x[i:i+lw]*window))
	return res

def automeanwave(x, fs):
	#f0
	f0 = fundamental_frequency(x, fs)
	print(f0)
	win_size = int((fs/f0)*1.2)

	randWin_index = rd.randint(0, len(x)-win_size)
	win = x[randWin_index:randWin_index+win_size]
	res = sumvolve(x, win)

	return res, win

def plotfft(s, fmax, doplot=False):
	""" This functions computes the fft of a signal, returning the frequency
	and their magnitude values.

	Parameters
	----------
	s: array-like
	  the input signal.
	fmax: int
	  the sampling frequency.
	doplot: boolean
	  a variable to indicate whether the plot is done or not.

	Returns
	-------
	f: array-like
	  the frequency values (xx axis)
	fs: array-like
	  the amplitude of the frequency values (yy axis)
	"""

	fs = abs(np.fft.fft(s))
	f = np.linspace(0, fmax // 2, len(s) // 2)
	if doplot:
		plt.plot(f[1:len(s) // 2], fs[1:len(s) // 2])
	return (f[1:len(s) // 2].copy(), fs[1:len(s) // 2].copy())


def fundamental_frequency(s, FS):
	# TODO: review fundamental frequency to guarantee that f0 exists
	# suggestion peak level should be bigger
	# TODO: explain code
	"""Compute fundamental frequency along the specified axes.

	Parameters
	----------
	s: ndarray
		input from which fundamental frequency is computed.
	FS: int
		sampling frequency
	Returns
	-------
	f0: int
	   its integer multiple best explain the content of the signal spectrum.
	"""

	s = s - np.mean(s)
	f, fs = plotfft(s, FS, doplot=False)

	# fs = smooth(fs, 50.0)

	fs = fs[1:len(fs) // 2]
	f = f[1:len(f) // 2]

	cond = np.where(f > 0.5)[0][0]

	print(cond)
	print(fs[cond:])

	bp = BigPeaks(fs[cond:], 0)

	if bp == []:
		f0 = 0
	else:

		bp = bp + cond

		f0 = f[min(bp)]

	return f0

def BigPeaks(s, th, min_peak_distance=5, peak_return_percentage=0.1):
	p = peaks(s, th)
	pp = []
	if len(p) == 0:
		pp = []
	else:
		p = clean_near_peaks(s, p, min_peak_distance)

		if len(p) != 0:
			ars = argsort(s[p])
			pp = p[ars]

			num_peaks_to_return = int(ceil(len(p) * peak_return_percentage))

			pp = pp[-num_peaks_to_return:]
		else:
			pp == []
	return pp

def chunk_data_str(data,window_size,overlap_size=0,flatten_inside_window=True):
	assert data.ndim == 1 or data.ndim == 2
	if data.ndim == 1:
		data = data.reshape((-1, 1))

	# get the number of overlapping windows that fit into the data
	num_windows = (data.shape[0] - window_size) // (window_size - overlap_size) + 1
	overhang = data.shape[0] - (num_windows * window_size - (num_windows - 1) * overlap_size)

	# if there's overhang, need an extra window and a zero pad on the data
	# (numpy 1.7 has a nice pad function I'm not using here)
	if overhang != 0:
		num_windows += 1
		newdata = np.empty((num_windows * window_size - (num_windows - 1) * overlap_size, data.shape[1])).astype(str)
		newdata[:data.shape[0]] = data
		data = newdata

	sz = data.dtype.itemsize
	ret = ast(
		data,
		shape=(num_windows, window_size * data.shape[1]),
		strides=((window_size - overlap_size) * data.shape[1] * sz, sz)
	)

	if flatten_inside_window:
		return ret
	else:
		return ret.reshape((num_windows, -1, data.shape[1]))


def chunk_data(data,window_size,overlap_size=0,flatten_inside_window=True):

	assert data.ndim == 1 or data.ndim == 2
	if data.ndim == 1:
		data = data.reshape((-1,1))

	# get the number of overlapping windows that fit into the data
	num_windows = (data.shape[0] - window_size) // (window_size - overlap_size) + 1
	overhang = data.shape[0] - (num_windows*window_size - (num_windows-1)*overlap_size)

	# if there's overhang, need an extra window and a zero pad on the data
	# (numpy 1.7 has a nice pad function I'm not using here)
	if overhang != 0:
		num_windows += 1
		newdata = np.zeros((num_windows*window_size - (num_windows-1)*overlap_size,data.shape[1]))
		newdata[:data.shape[0]] = data
		data = newdata

	sz = data.dtype.itemsize
	ret = ast(
			data,
			shape=(num_windows,window_size*data.shape[1]),
			strides=((window_size-overlap_size)*data.shape[1]*sz, sz)
			)

	if flatten_inside_window:
		return ret
	else:
		return ret.reshape((num_windows,-1,data.shape[1]))

def tsfelMulti(signal, fs):
	feat_dict = dict(temp_centroid = mpt.calc_centroid(signal, fs),
					 temp_maxpks = mpt.maxpeaks(signal),
					 temp_minpks = mpt.minpeaks(signal),
					 temp_meandiff = mpt.mean_diff(signal),
					 temp_mean_absdiff = mpt.mean_abs_diff(signal),
					 temp_median_absdiff = mpt.median_abs_diff(signal),
					 temp_dist = mpt.distance(signal),
					 temp_sumabsdiff = mpt.sum_abs_diff(signal),
					 temp_zcr = mpt.zero_cross(signal),
					 temp_tenergy = mpt.total_energy(signal, fs),
					 temp_mslope = mpt.mean_slope(signal),
					 temp_auc = mpt.auc(signal, fs),
					 temp_abs_energy = mpt.abs_energy(signal),
					 temp_pk2pk = mpt.pk_pk_distance(signal),
					 stat_interq = mpt.interq_range(signal),
					 stat_kurt = mpt.kurtosis(signal),
					 stat_ske = mpt.skewness(signal),
					 stat_c_max = mpt.calc_max(signal),
					 stat_c_min = mpt.calc_min(signal),
					 stat_c_mean = mpt.calc_mean(signal),
					 stat_c_median = mpt.calc_median(signal),
					 stat_c_std = mpt.calc_std(signal),
					 stat_c_var = mpt.calc_var(signal),
					 stat_m_abs_dev = mpt.mean_abs_deviation(signal),
					 stat_med_abs_dev = mpt.median_abs_deviation(signal),
					 stat_rms_s = mpt.rms(signal),
					 spec_s_dist = mpt.spectral_distance(signal, fs=fs),
					 spec_f_f = mpt.fundamental_frequency(signal, fs=fs),
					 spec_m_freq = mpt.max_frequency(signal, fs),
					 spec_kurt = mpt.spectral_kurtosis(signal, fs),
					 spec_skew = mpt.spectral_skewness(signal, fs),
					 spec_spread = mpt.spectral_spread(signal, fs),
					 spec_roff = mpt.spectral_roll_off(signal, fs),
					 spec_ron = mpt.spectral_roll_on(signal, fs),
					 spec_entropy = mpt.spectral_entropy(signal, fs),
					 spec_m_coeff = mpt.fft_mean_coeff(signal, fs))


	return feat_dict


def featuresTsfelMat(inputSignal, fs, window_len):

	inputSignal = inputSignal - np.mean(inputSignal)
	WinRange = int(window_len / 2)
	t = np.linspace(0, len(inputSignal[:,0])/fs, len(inputSignal[:,0]))
	t = chunk_data(np.r_[t[WinRange:0:-1], t, t[-1:len(t) - WinRange:-1]], window_size=window_len, overlap_size=window_len-1)

	if(np.ndim(inputSignal)>1):
		s_matrix = []
		sig = np.r_[inputSignal[WinRange:0:-1,:], inputSignal, inputSignal[-1:len(inputSignal) - WinRange:-1,:]].transpose()
		for s_i in range(np.shape(sig)[0]):
			s_t = sig[s_i]
			s_temp = np.copy(s_t)
			sig_a = chunk_data(s_temp, window_size=window_len, overlap_size=window_len - 1)
			s_matrix.append(sig_a)
	else:
		sig = np.r_[inputSignal[WinRange:0:-1], inputSignal, inputSignal[-1:len(inputSignal) - WinRange:-1]].transpose()
		s_temp = np.copy(sig)
		sig_a = chunk_data(s_temp, window_size=window_len, overlap_size=window_len - 1)
		s_matrix = sig_a

	t1 = time.time()
	output = np.array([{"signal":in_sig_i, "features": tsfelMulti(sig_i, fs)} for sig_i, in_sig_i in zip(s_matrix, inputSignal)])
	# s222 = np.array([mpt.fft_mean_coeff(sig_i, fs) for sig_i in s_matrix])

	t2 = time.time()

	# mpt.calc_centroid
	# mpt.max_pks (corrigir threshold da derivada)
	# mpt.min_pks (mesmo que o anterior)
	# mpt.mean_diff
	# mpt.mean_abs_diff
	# mpt.median_diff
	# mpt.median_abs_diff
	# distance
	# sum_abs_diff
	# zero_cross
	# total_energy
	# mean_slope
	# auc
	# abs_energy
	# pk2pk
	# interq_range
	# all stats except for ecdf
	# spectral_kurtosis
	# spectral_spread
	# spectral_skew
	# spectral roll_off
	# spectral roll_on
	# spectral_entropy
	# fft_mean_coeff
	# no_wavelet

	"""TODO: rever questoes do signal[0] quando for apenas 1 sinal
		     rever questoes da derivada
	"""

	print(t2-t1)
	#
	# key=12
	#
	# # for key in range(0, 21):
	# sig2 = np.r_[inputSignal[WinRange:0:-1, key], inputSignal[:, key], inputSignal[-1:len(inputSignal[:,key]) - WinRange:-1, key]]
	# # plt.plot(sig2)
	# # print(len(inputSignal[:, key]))
	# output = np.zeros(len(inputSignal[:, key]))
	#
	# for i in range(int(WinRange), len(sig2) - int(WinRange)):
	# 	signal_segment = sig2[i - WinRange:WinRange + i]
	# 	# plt.plot(signal_segment)
	# 	# plt.show()
	# 	output[i - int(WinRange)] = tsfel_ft.fft_mean_coeff(signal_segment, fs)

	# plt.plot(inputSignal[:,key]/np.max(inputSignal[:,key]))
	# plt.plot(output)
	# plt.plot(s222.transpose()[:, key])
	# plt.show()
	return output

def featuresTsfel(inputSignal, fs, window_len, window="hanning", domain="temporal"):

	win = eval('np.' + window + '(window_len)')

	inputSignal = inputSignal - np.mean(inputSignal)

	WinRange = int(window_len / 2)

	sig = np.r_[inputSignal[WinRange:0:-1], inputSignal, inputSignal[-1:len(inputSignal) - WinRange:-1]]

	if domain is "all":
		feats = tsfel.feature_extraction.get_features_by_domain()
	elif(domain is "temporal"):
		feats = tsfel.feature_extraction.get_features_by_domain("temporal")
	elif(domain is "statistical"):
		feats = tsfel.feature_extraction.get_features_by_domain("statistical")
	elif(domain is "spectral"):
		feats = tsfel.feature_extraction.get_features_by_domain("spectral")
	else:
		feats = 0
		print("domain is not recognized")

	data = {}

	for dom in feats.keys():
		print("Initiating extraction of "+dom+" features")
		data[dom] = {}
		lst_ks = list(feats[dom].keys())

		output = np.zeros((len(inputSignal), len(lst_ks)))

		for feat_i, feat in enumerate(lst_ks):
			parameters = feats[dom][feat]["parameters"]
			if (len(parameters) < 1):
				for i in range(int(WinRange), len(sig) - int(WinRange)):
					signal_segment = sig[i - WinRange:WinRange + i]
					output[i - int(WinRange), feat_i] = eval(feats[dom][feat]["function"]+"(signal_segment)")
					# else:
					#     if("fs" in list(parameters.keys())):
					#         print(feats[dom][feat]["parameters"])
					#         output[i - int(WinRange), feat_i] = eval(feats[dom][feat]["function"] + "(signal_segment, fs=fs)")
			data[dom][feat] = output[:, feat_i]

	return data


def NewWindowStat(inputSignal, statTools, fs, window_len=50, window="hanning"):

	win = eval('np.' + window + '(window_len)')

	if inputSignal.ndim != 1:
		raise ValueError("smooth only accepts 1 dimension arrays.")
	if inputSignal.size < window_len:
		raise ValueError("Input vector needs to be bigger than window size.")
	if window_len < 3:
		return inputSignal

	inputSignal = inputSignal - np.mean(inputSignal)

	WinRange = int(window_len / 2)

	sig = np.r_[inputSignal[WinRange:0:-1], inputSignal, inputSignal[-1:len(inputSignal) - WinRange:-1]]

	if(len(statTools)>1):

		output = np.zeros((len(inputSignal), len(statTools)))

		for i in range(int(WinRange), len(sig) - int(WinRange)):

			for n, statTool in enumerate(statTools):

				signal_segment = sig[i - WinRange:WinRange + i]
				if(statTool is "zcr"):
					output[i - int(WinRange), n] = ZeroCrossingRate(signal_segment*win)
				elif(statTool is 'Azcr'):
					A = np.max(signal_segment)
					output[i - int(WinRange), n] = A*ZeroCrossingRate(signal_segment)
				elif(statTool is "std"):
					output[i - int(WinRange), n] = np.std(signal_segment*win)
				elif(statTool is "skw"):
					output[i - int(WinRange), n] = skew(signal_segment)
				elif (statTool is "mean"):
					output[i - int(WinRange), n] = np.mean(signal_segment*win)
				elif (statTool is "subPks"):
					pks = [0]
					win_len = window_len
					while (len(pks) < 10):
						pks = detect_peaks(signal_segment, valley=False,
										   mph=np.std(signal_segment))
						if (len(pks) < 10):
							win_len += int(win_len / 5)
					sub_zero = pks[1] - pks[0]
					sub_end = pks[-1] - pks[-2]
					subPks = np.r_[sub_zero, (pks[1:-1] - pks[0:-2]), sub_end]
					output[i - int(WinRange), n] = np.mean(subPks)

				elif (statTool is "findPks"):
					pks = detect_peaks(signal_segment*win, valley=False,
									   mph=np.std(signal_segment))
					LenPks = len(pks)
					output[i - int(WinRange), n] = LenPks

				elif(statTool is 'sum'):
					output[i - WinRange, n] = np.sum(abs(signal_segment)*win)
				elif(statTool is 'normal'):
					output[i - WinRange, n] = normaltest(signal_segment)[0]
				elif(statTool is 'krt'):
					output[i - WinRange, n] = kurtosis(signal_segment)

				elif (statTool is 'AmpDiff'):
					maxPks = detect_peaks(signal_segment*win, valley=False,
										  mph=np.std(signal_segment))
					minPks = detect_peaks(signal_segment*win, valley=True,
										  mph=np.std(signal_segment))
					AmpDiff = np.sum(signal_segment[maxPks]) - np.sum(signal_segment[minPks])
					output[i - WinRange] = AmpDiff

				elif (statTool is "SumPS"):
					f, Pxx = PowerSpectrum(signal_segment*win, fs=fs, nperseg=WinRange / 2)
					sps = SumPowerSpectrum(Pxx)
					output[i - WinRange, n] = sps

				elif (statTool is "AmpMean"):
					output[i - WinRange, n] = np.mean(abs(signal_segment))

				elif (statTool is "Spikes1"):
					ss = 0.1 * max(sig)
					pkd, md = Spikes(signal_segment, mph=ss)
					output[i - WinRange, n] = pkd

				elif (statTool is "Spikes2"):
					ss = 0.1 * max(sig)
					pkd, md = Spikes(signal_segment, mph=ss)
					output[i - WinRange, n] = md

				elif (statTool is "Spikes3"):
					ss = 0.1 * max(sig)
					pkd, md = Spikes(abs(signal_segment), mph=ss)
					output[i - WinRange, n] = md

				# elif (statTool is "df"):


		# output = output - np.mean(output)
		# output = output / max(output)

	else:
		output = WindowStat(inputSignal, statTools[0], fs, window_len=50, window='hanning')

	return output

#WindMethod
def WindowStat(inputSignal, statTool, fs, window_len=50, window='hanning'):

	output = np.zeros(len(inputSignal))
	win = eval('np.' + window + '(window_len)')

	if inputSignal.ndim != 1:
		raise ValueError("smooth only accepts 1 dimension arrays.")
	if inputSignal.size < window_len:
		raise ValueError("Input vector needs to be bigger than window size.")
	if window_len < 3:
		return inputSignal

	inputSignal = inputSignal - np.mean(inputSignal)

	WinRange = int(window_len/2)

	sig = np.r_[inputSignal[WinRange:0:-1], inputSignal, inputSignal[-1:len(inputSignal)-WinRange:-1]]

	# windowing
	if(statTool is 'stn'):
		WinSize = window_len
		numSeg = int(len(inputSignal) / WinSize)
		SigTemp = np.zeros(numSeg)
		for i in range(1, numSeg):
			signal = inputSignal[(i - 1) * WinSize:i * WinSize]
			SigTemp[i] = sc.signaltonoise(signal)
		output = np.interp(np.linspace(0, len(SigTemp), len(output)), np.linspace(0, len(SigTemp), len(SigTemp)), SigTemp)
	elif(statTool is 'zcr'):
		# inputSignal = inputSignal - smooth(inputSignal, window_len=fs*4)
		# inputSignal = inputSignal - smooth(inputSignal, window_len=int(fs/10))
		# sig = np.r_[inputSignal[WinRange:0:-1], inputSignal, inputSignal[-1:len(inputSignal) - WinRange:-1]]

		for i in range(int(WinRange), len(sig) - int(WinRange)):
			output[i - int(WinRange)] = ZeroCrossingRate(sig[i - WinRange:WinRange + i]*win)
		output = smooth(output, window_len=1024)
	elif (statTool is 'Azcr'):
		# inputSignal = inputSignal - smooth(inputSignal, window_len=fs*4)
		# inputSignal = inputSignal - smooth(inputSignal, window_len=int(fs/10))
		# sig = np.r_[inputSignal[WinRange:0:-1], inputSignal, inputSignal[-1:len(inputSignal) - WinRange:-1]]

		for i in range(int(WinRange), len(sig) - int(WinRange)):
			A = np.max(sig[i - WinRange:WinRange + i])
			output[i - int(WinRange)] = A * ZeroCrossingRate(sig[i - WinRange:WinRange + i] * win)
		output = smooth(output, window_len=1024)
	elif(statTool is 'std'):
		for i in range(int(WinRange), len(sig) - int(WinRange)):
			output[i - WinRange] = np.std(sig[i - WinRange:WinRange + i]*win)
	elif (statTool is 'mean'):
		for i in range(int(WinRange), len(sig) - int(WinRange)):
			output[i - WinRange] = np.mean(sig[i - WinRange:WinRange + i])
	elif(statTool is 'subPks'):
		for i in range(int(WinRange), len(sig) - int(WinRange)):
			pks = [0]
			win_len = window_len
			while(len(pks) < 10):
				pks = detect_peaks(sig[i - int(win_len / 2):int(win_len / 2) + i], valley=False, mph=np.std(sig[i - int(win_len / 2):int(win_len / 2)+ i]))
				if(len(pks) < 10):
					win_len += int(win_len/5)
			sub_zero = pks[1] - pks[0]
			sub_end = pks[-1] - pks[-2]
			subPks = np.r_[sub_zero, (pks[1:-1] - pks[0:-2]), sub_end]
			win = eval('np.' + window + '(len(subPks))')
			output[i - int(WinRange)] = np.mean(subPks*win)
	elif (statTool is 'findPks'):
		for i in range(int(WinRange), len(sig) - int(WinRange)):
			pks = detect_peaks(sig[i - WinRange:WinRange + i], valley=False,
								   mph=np.std(sig[i - WinRange:WinRange + i]))
			LenPks = len(pks)
			output[i - int(WinRange)] = LenPks
	elif(statTool is 'sum'):
		for i in range(int(WinRange), len(sig) - int(WinRange)):
			output[i - WinRange] = np.sum(abs(sig[i - WinRange:WinRange + i] * win))
	elif(statTool is 'AmpDiff'):
		for i in range(int(WinRange), len(sig) - int(WinRange)):
			win_len = window_len
			tempSig = sig[i - int(win_len / 2):int(win_len / 2) + i]
			maxPks = detect_peaks(tempSig, valley=False,
								   mph=np.std(tempSig))
			minPks = detect_peaks(tempSig, valley=True,
								   mph=np.std(tempSig))
			AmpDiff = np.sum(tempSig[maxPks]) - np.sum(tempSig[minPks])
			output[i - WinRange] = AmpDiff
	elif(statTool is 'MF'):
		for i in range(int(WinRange), len(sig) - int(WinRange)):
			f, Pxx = PowerSpectrum(inputSignal[i - WinRange:i + WinRange], fs=fs, nperseg=WinRange/2)
			mf = MF_calculus(Pxx)
			output[i - WinRange] = mf
	elif(statTool is "SumPS"):
		for i in range(int(WinRange), len(sig) - int(WinRange)):
			f, Pxx = PowerSpectrum(inputSignal[i - WinRange:i + WinRange], fs=fs, nperseg=WinRange / 2)
			sps = SumPowerSpectrum(Pxx)
			output[i - WinRange] = sps
	elif(statTool is "AmpMean"):
		for i in range(int(WinRange), len(sig) - int(WinRange)):
			output[i - WinRange] = np.mean(abs(sig[i - WinRange:WinRange + i]) * win)
	elif(statTool is"Spikes1"):
		ss = 0.1*max(sig)
		for i in range(int(WinRange), len(sig) - int(WinRange)):
			pkd, md = Spikes(sig[i - WinRange:WinRange + i] * win, mph=ss)
			output[i - WinRange] = pkd
	elif (statTool is "Spikes2"):
		ss = 0.1 * max(sig)
		for i in range(int(WinRange), len(sig) - int(WinRange)):
			pkd, md = Spikes(sig[i - WinRange:WinRange + i] * win, mph=ss)
			output[i - WinRange] = md
	elif (statTool is "Spikes3"):
		ss = 0.1 * max(sig)
		for i in range(int(WinRange), len(sig) - int(WinRange)):
			pkd, md = Spikes(abs(sig[i - WinRange:WinRange + i] )* win, mph=ss)
			output[i - WinRange] = md

	output = output - np.mean(output)
	output = output/max(output)
	#output = smooth(output, window_len=10)

	return output

def ZeroCrossingRate(signal):
	signal = signal - np.mean(signal)
	ZCVector = np.where(np.diff(np.sign(signal)))[0]

	return len(ZCVector)

def findPeakDistance(signal, mph, threshold):
	pks = detect_peaks(signal, mph = mph, show = False)
	vpks = detect_peaks(signal, mph= mph, valley=True)

	if(len(vpks)> len(pks)):
		pks = vpks

	signaldPks = np.zeros(np.size(signal))
	dpks = np.log10(abs(np.diff(pks) - np.mean(np.diff(pks))) + 1)

	for i in range(0, len(dpks)):
		if(i == 0):
			signaldPks[0:pks[i]] = dpks[i]
			signaldPks[pks[i]:pks[i + 1]] = dpks[i]
		elif(i == len(dpks)-1):
			signaldPks[pks[i]:pks[i+1]] = dpks[-1]
		else:
			signaldPks[pks[i]:pks[i+1]] = dpks[i]

def MF_calculus(Pxx):
	sumPxx = np.sum(Pxx)
	mf = 0
	for i in range(0, len(Pxx)):
		if(np.sum(Pxx[0:i]) < sumPxx/2.0):
			continue
		else:
			mf = i
			break

	return mf

def SumPowerSpectrum(Pxx):
	return np.sum(Pxx)

def PowerSpectrum(data, fs, nperseg):
	f, Pxx = sc.periodogram(data, fs=fs)

	return f, Pxx

def Spikes(inputSignal, mph, edge="rising"):
	pks = detect_peaks(inputSignal, mph=mph)
	numPics = len(pks)
	if(len(pks)<2):
		meanDistance=0
	else:
		meanDistance = np.mean(np.diff(pks))

	return numPics, meanDistance

def mean_norm(sig):
	a = sig-np.mean(sig)
	return a/max(a)

def notchFilter(s, f, fs, Q):

	b,a = sc.iirnotch(f, Q, fs)

	return sc.filtfilt(b, a, s)

