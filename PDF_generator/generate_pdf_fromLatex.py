import argparse
import os
import subprocess
import matplotlib.pyplot as plt
from tools.load_tools import loadH5
from tools.plot_tools import *
import pdflatex


example_path = "/media/jeanraltique/FishStory/Projectos/Doutoramento/PhDCode/PhDProject/Hui_SuperProject/Data_Examples/"

signal = loadH5(example_path+"arthrokinemat_2018_06_02_23_51_55.h5")
fs = 1000
b = 10
ch1 = signal[b*fs:, 5]

#pre process
ch1_raw = ch1

plt.plot(ch1_raw)
plt.savefig('example.png')
close()



content = r'''

\documentclass{article}
\usepackage{graphicx}
\begin{document}
...
\textbf{\huge %(school)s \\}
\vspace{1cm}
\textbf{\Large %(title)s \\}

\begin{figure}
	\includegraphics[width=15cm]{example.jpg}
	\caption{Example figure proudly made with matplotlib's PGF backend.}
\end{figure}
...
\end{document}
'''

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--course')
parser.add_argument('-t', '--title', default = "Hello World")
parser.add_argument('-n', '--name')
parser.add_argument('-s', '--school', default='My U')

args = parser.parse_args()

with open('cover.tex','w') as f:
    f.write(content%args.__dict__)

with open('cover.tex', 'rb') as f:
    pdfl = pdflatex.PDFLaTeX.from_binarystring(f.read(), 'cover')

pdfl.set_output_directory("/media/jeanraltique/FishStory/Projectos/Doutoramento/PhDCode/PhDProject/PDF_generator")
pdfl.set_pdf_filename("cover")


pdf, log, cp = pdfl.create_pdf(keep_pdf_file=True)
