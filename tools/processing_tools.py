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

	return np.apply_along_axis("".join, axis, mat_string)

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
					output[i - int(WinRange), n] = ZeroCrossingRate(signal_segment)
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
				elif (statTool is 'normal'):
					output[i - WinRange, n] = normaltest(signal_segment)[0]
				elif (statTool is 'krt'):
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
			output[i - WinRange] = np.mean(sig[i - WinRange:WinRange + i] * win)
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

