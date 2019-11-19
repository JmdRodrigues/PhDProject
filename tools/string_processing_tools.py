from itertools import groupby
import numpy as np
from GrammarofTime.SSTS.backend.gotstools import levenshteinDist
from collections import OrderedDict
import regex as re
import textdistance
from textblob import TextBlob


def runLengthEncoding(list_str):
    # Generate ordered dictionary of all lower
    # case alphabets, its output will be
    # dict = {'w':0, 'a':0, 'd':0, 'e':0, 'x':0}

    count = 1
    previous = ""
    counts = []
    words = []


    for word in list_str:
        if word != previous:
            if previous:
                counts.append(count)
                words.append(previous)
            count = 1
            previous = word
        else:
            count += 1
    else:
        counts.append(count)
        words.append(previous)

    result_join = " ".join(f"{str(count)}{word}" for word, count in zip(words, counts))


    return result_join, words, counts




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


def Ngrams(strSeq, n=2):
    strSeq_blob = TextBlob(" ".join([word for word in strSeq]))
    seq_grams = strSeq_blob.ngrams(n)

    grammed_words = ["".join([w for w in sentence]) for sentence in seq_grams]

    grammed_strSeq = [strSeq[i] + strSeq[i+1] for i in range(0, len(strSeq)-1)]

    return grammed_words, grammed_strSeq



def BagofWords(strSeq):
    for sample in set(strSeq):
        print(sample)
        print(strSeq.count(sample))
    return dict((sample, strSeq.count(sample)) for sample in set(strSeq))

def tf(strDict, unique_set):
    """

    :param strDict: Bag of words dict of the sequence
    :return: tf for each element of the dict
    """

    nk = sum(strDict.values())
    print(strDict)
    strDict_tf = {}

    for word in unique_set:
        if(word in list(strDict.keys())):
            strDict_tf[word] = {"freq":strDict[word], "tf":strDict[word]/nk, "idf_cnt":1}

        else:
            strDict_tf[word] = {"freq":0, "tf":0, "idf_cnt":0}

    return strDict_tf

def idf(strings_list):
    n = len(strings_list)
    idf = dict.fromkeys(strings_list[0].keys(), 0)
    for l in strings_list:
        for word, count in l.items():
            if count > 0:
                idf[word] += 1

    for word, v in idf.items():
        idf[word] = np.log(n / float(v))

    return idf



def tf_idf(strDict, unique_set):
    """

    :param strDict: Dict with doc as keys and word list as values {doc1:["a", "a", "b"], doc2:[...}
    :return: tf_idf for each word of each doc, and tf for each word of each doc
    """

    keys = list(strDict.keys())
    n_docs = len(keys)

    idf = dict.fromkeys(unique_set, 0)
    tf_Dict = {doc:{} for doc in keys}
    tf_idf_Dict = {doc:{} for doc in keys}

    for doc_i in keys:

        #bag of words of doc
        BoW = BagofWords(strDict[doc_i])
        #tf of doc
        tf_Dict[doc_i] = tf(BoW, unique_set)

        for word in unique_set:
            if word in list(BoW.keys()):
                idf[word] += 1

    for word, v in idf.items():
        idf[word] = np.log(n_docs / float(v))

        for doc in keys:
            tf_idf_Dict[doc][word] = tf_Dict[doc][word]["tf"]*idf[word]

    return tf_idf_Dict, tf_Dict

def Decode_tfDict(strDict, tf_Dict):
    keys = list(strDict.keys())
    n_docs = len(keys)

    strDecode = {doc_i:[] for doc_i in keys}
    for doc_i in keys:

        tf = tf_Dict[doc_i]
        strDecode[doc_i] = [tf[word]["tf"] for word in strDict[doc_i]]

    return strDecode


def Seq_StringDistance(str_seq, str_ref, method="hamming"):

    if(method is "hamming"):
        return [textdistance.hamming(str_seq_i, str_ref) for str_seq_i in str_seq]

    elif(method is "levenshtein"):
        return [textdistance.levenshtein(str_seq_i, str_ref) for str_seq_i in str_seq]

    elif (method is "damerau_lev"):
        return [textdistance.damerau_levenshtein(str_seq_i, str_ref) for str_seq_i in str_seq]

    elif (method is "j-winkler"):
        return [textdistance.jaro_winkler(str_seq_i, str_ref) for str_seq_i in str_seq]

    elif (method is "smith-waterman"):
        return [textdistance.smith_waterman(str_seq_i, str_ref) for str_seq_i in str_seq]

    elif (method is "jaccard"):
        return [textdistance.jaccard(str_seq_i, str_ref) for str_seq_i in str_seq]

    elif (method is "sorensen-dice"):
        return [textdistance.sorensen_dice(str_seq_i, str_ref) for str_seq_i in str_seq]

    elif (method is "tversky"):
        return [textdistance.tversky(str_seq_i, str_ref) for str_seq_i in str_seq]

    elif (method is "tanimoto"):
        return [textdistance.tanimoto(str_seq_i, str_ref) for str_seq_i in str_seq]

    elif (method is "cosine"):
        return [textdistance.cosine(str_seq_i, str_ref) for str_seq_i in str_seq]

    elif (method is "tanimoto"):
        return [textdistance.tanimoto(str_seq_i, str_ref) for str_seq_i in str_seq]

    elif (method is "ratcliff"):
        return [textdistance.ratcliff_obershelp(str_seq_i, str_ref) for str_seq_i in str_seq]

    elif (method is "bwt"):
        return [textdistance.bwtrle_ncd(str_seq_i, str_ref) for str_seq_i in str_seq]

def mostFreqSeq(strSeq):

    return max(set(strSeq), key = strSeq.count)

def leastFreqSeq(strSeq):

    return min(set(strSeq), key=strSeq.count)