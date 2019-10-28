from jinja2 import Environment, FileSystemLoader
import pdfkit
import weasyprint
import numpy as np
import matplotlib.pyplot as plt

file_loader = FileSystemLoader("/media/jeanraltique/FishStory/Projectos/Doutoramento/PhDCode/PhDProject/PDF_generator/testJinja2")
env = Environment(loader=file_loader)

a = np.random.randint(0, 100, 100)
t = np.linspace(0, 1, 100)

fig = plt.plot(t, a)
plt.savefig("example.png")

template = env.get_template("template_test.html")
output = template.render(title="Best Title", graph="""example.png""")
print(output)
html = open("output.html", "w")
html.write(output)
html.close()
import os

html = weasyprint.HTML('output.html')

pdf = html.write_pdf("output.pdf")
#
# from pathlib import Path
#
# p = Path(Path.home()).parents[1]
#
#
# path_wkhtmltopdf = "usr/bin/"
# config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)
# pdfkit.from_file("/media/jeanraltique/FishStory/Projectos/Doutoramento/PhDCode/PhDProject/PDF_generator/testJinja2/output.html", "test.pdf", configuration=config)