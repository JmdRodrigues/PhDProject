from GrammarofTime.SSTS.backend import gotstools as gt
from tools.load_tools import loadH5
from Hui_SuperProject.parser_reader import read_string
from tools.processing_tools import *
from tools.string_processing_tools import *
import matplotlib.pyplot as plt
from novainstrumentation import niplot, multilineplot
from tools.plot_tools import *
import regex as re
from tsfel.feature_extraction.features import *
from novainstrumentation import sumvolve
from novainstrumentation.freq_analysis import fundamental_frequency


example_path = r"C:\Users\Wolfgang\PycharmProjects\ppp\Hui_SuperProject\Data_Examples\\"

signal = loadH5(example_path+"arthrokinemat_2018_06_03_00_06_57.h5")

fs = 1000
b = 10
acc1 = signal[b*fs:, 5]




