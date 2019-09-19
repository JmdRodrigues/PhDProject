from itertools import groupby
import numpy as np
from GrammarofTime.SSTS.backend.gotstools import levenshteinDist
import regex as re

def consecutiveCount(s, type="numbered"):
    groups = groupby(s)
    if(type is "numbered"):
        result = [label + str(sum(1 for _ in group)) for label, group in groups]
        return "".join(result)
    elif(type is "grouped"):
        result = [(label, sum(1 for _ in group)) for label, group in groups]
        return result
    else:
        result = [label for label, group in groups]
        groups = groupby(s)
        indx = [sum(1 for _ in group) for label, group in groups]

        return "".join(result), indx

def findStringIndexes(string, substring, consecutive_count):

    indxs =  [m.start() for m in re.finditer(substring, string)]
    print(indxs)
    true_indxs = [np.sum(consecutive_count[:indx]) for indx in indxs]
    print(true_indxs)

    return true_indxs


def CountSequences(ref_s, substrings):
    freq = []
    for str_i in substrings:
        freq.append(ref_s.count(str_i))

    return freq
def CountSeqLeveled(ref_s, initial_substrings, level):

    substrings = initial_substrings
    # for i in range(level):
    freq = []
    for str_i in substrings:
        freq.append(ref_s.count(str_i))
    #create new set of substrings
    sorted_substrings = sorted(zip(freq, substrings))

    # most frequent char
    mf_c = substrings[np.argmax(freq)]

    for i in range(level):
        #get max indexes where last char is present as a beginning in substring
        l_char = max([(freq[substrings.index(s)], s) for s in substrings if mf_c[-1] == s[0]])

        #add it to the previous string
        mf_c += l_char[1][-1]

        freq.pop(substrings.index(l_char[1]))
        substrings.remove(l_char[1])

    return mf_c



def WindowString(inputString, substring, statTool, window_len=50):

	output = np.zeros(len(inputString))

	WinRange = int(window_len/2)

	stringS = inputString[WinRange:0:-1] + inputString + inputString[-1:len(inputString)-WinRange:-1]

	# stringS = np.r_[inputString[WinRange:0:-1], inputString, inputString[-1:len(inputString)-WinRange:-1]]

	# windowing
	if(statTool is 'levenshtein'):
		for i in range(int(WinRange), len(stringS) - int(WinRange)):
			output[i - int(WinRange)] = levenshteinDist(stringS[i - WinRange:WinRange + i], substring)

	return output
