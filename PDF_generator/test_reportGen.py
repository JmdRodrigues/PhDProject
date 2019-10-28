import PDF_generator.reportGen as rg
import matplotlib.pyplot as plt
import numpy as np

a = np.random.randint(0, 10, 1000)

fig = plt.figure()
plt.plot(a)

doc = rg.Report("libphys", "test_report")
doc.add_title("My first figure")
doc.add_graph(fig, "fig_test")
mkd_content = """

##Adding some Markdown for *fun*

#### Welcome to StackEdit!

Hi! I'm your first Markdown file in **StackEdit**. If you want to learn about StackEdit, you can read me. If you want to play with Markdown, you can edit me. Once you have finished with me, you can create new files by opening the **file explorer** on the left corner of the navigation bar.


#### Files

StackEdit stores your files in your browser, which means all your files are automatically saved locally and are accessible **offline!**
"""
doc.add_MKD(mkd_content)
doc.add_MKD(mkd_content)

doc.gen_pdf()